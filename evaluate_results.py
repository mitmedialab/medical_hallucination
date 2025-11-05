"""
Medical Hallucination Evaluation Script

This script evaluates the results from medical hallucination mitigation experiments.
It calculates:
- Accuracy scores for each mitigation method
- Pointwise scores (correct=+1, incorrect=-0.25)
- Similarity scores using UMLS-BERT embeddings
- Answer similarity (model output vs correct answer)
- Question similarity (model output vs question)
- Combined similarity score

Usage:
    python evaluate_results.py --results_dir ./results --models gpt-4o gemini-2.0-flash --tasks FCT fake nota

For more details, see README.md
"""

import torch
import numpy as np
import re
import json
import os
import pandas as pd
import argparse
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


class UMLSBERT:
    """UMLS-BERT encoder for medical text embeddings"""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts):
        """Encode texts to embeddings"""
        new_texts = []
        for t in texts:
            if isinstance(t, dict):
                new_texts.append(t.get('error', str(t)))
            else:
                new_texts.append(t)

        texts = new_texts
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            model_output = self.model(**inputs)
            attention_mask = inputs["attention_mask"]
            embeddings = self.mean_pooling(model_output, attention_mask)

        return embeddings.cpu().numpy()


class MedicalAnalyzer:
    """Analyzer for medical responses using embeddings"""
    def __init__(self):
        self.umlsbert = UMLSBERT()

    def analyze_response(self, question_data: Dict) -> Dict:
        """Analyze medical responses using embeddings"""
        results = {}

        # Get base texts for comparison
        base_texts = {
            'question': question_data['question'],
            'correct_option': question_data['options']['A'],  # Assuming A is correct
            'outputs': {}
        }

        # Add available outputs and ensure they are strings
        for output_type in ['base_output', 'prompting_output', 'cot_output', 'medrag_output', 'internetsearch_output']:
            if output := question_data.get(output_type):
                # Convert non-string outputs to strings
                if isinstance(output, dict):
                    output = json.dumps(output)
                base_texts['outputs'][output_type] = str(output)

        # Calculate embeddings
        embeddings = {}
        all_texts = [base_texts['question'], base_texts['correct_option']]
        text_keys = ['question', 'correct_option']

        for output_type, text in base_texts['outputs'].items():
            all_texts.append(text)
            text_keys.append(output_type)

        all_embeddings = self.umlsbert.encode(all_texts)

        # Store embeddings in dictionary
        for idx, key in enumerate(text_keys):
            embeddings[key] = all_embeddings[idx]

        # Calculate similarities
        similarities = {}
        for output_type in base_texts['outputs'].keys():
            # Calculate similarity with correct answer
            similarity = float(cosine_similarity(
                [embeddings['correct_option']],
                [embeddings[output_type]]
            )[0][0])

            # Calculate similarity with question
            question_similarity = float(cosine_similarity(
                [embeddings['question']],
                [embeddings[output_type]]
            )[0][0])

            similarities[output_type] = {
                'answer_similarity': similarity,
                'question_similarity': question_similarity,
                'combined_score': (similarity + question_similarity) / 2
            }

        # Add metadata
        return {
            'similarities': similarities,
            'metadata': {
                'question_length': len(question_data['question'].split()),
                'correct_option_length': len(base_texts['correct_option'].split()),
                'output_lengths': {
                    k: len(v.split())
                    for k, v in base_texts['outputs'].items()
                }
            }
        }


def evaluate_medical_responses(results_dir: str, dataset_dir: str, models: List[str], tasks: List[str]) -> Dict:
    """Evaluate medical responses across different models and tasks"""
    model_performance = defaultdict(lambda: defaultdict(lambda: {
        "correct_sum": 0,
        "total_count": 0,
        "pointwise_score": 0,
        "similarity_scores": defaultdict(list),
        "all_scores": defaultdict(list)
    }))

    # Scoring parameters
    P_c = 1  # Points for correct prediction
    P_w = -0.25  # Points for incorrect prediction

    analyzer = MedicalAnalyzer()

    for model_name in models:
        for task in tasks:
            results_json_path = os.path.join(results_dir, f"{model_name}_medhalt_reasoning_{task}_seed0.json")
            reasoning_csv_path = os.path.join(dataset_dir, "original", f"reasoning_{task}.csv")

            if not os.path.exists(results_json_path):
                print(f"File not found: {results_json_path}")
                continue

            print(f"[INFO] Processing {model_name} - {task}")

            # Load data
            try:
                if task.lower() != 'fake':
                    if not os.path.exists(reasoning_csv_path):
                        print(f"Ground truth file not found: {reasoning_csv_path}")
                        continue

                    reasoning_df = pd.read_csv(reasoning_csv_path)
                    if not all(col in reasoning_df.columns for col in ["question", "correct_answer", "correct_index"]):
                        print(f"Missing required columns in CSV for {task}")
                        continue

                    question_to_correct = dict(zip(reasoning_df["question"],
                                                zip(reasoning_df["correct_answer"],
                                                    reasoning_df["correct_index"])))
                else:
                    # For 'fake' task, correct answer is always "I do not know"
                    with open(results_json_path, 'r') as f:
                        results_data = json.load(f)
                    question_to_correct = {
                        sample["question"]: ("I do not know", 5)
                        for sample in results_data.get('results', [])
                    }

            except Exception as e:
                print(f"Error processing {task}: {e}")
                continue

            with open(results_json_path, 'r') as f:
                results_data = json.load(f)

            # Process each sample
            for sample in results_data.get('results', []):
                options = sample.get('options', {})
                question = sample.get("question", "")
                if question not in question_to_correct:
                    continue

                correct_answer, correct_idx = question_to_correct[question]

                # Convert correct_idx to string format
                try:
                    if task.lower() != 'fake':
                        correct_idx = str(int(str(correct_idx)) + 1)  # 1-based index
                    else:
                        correct_idx = '6'
                except:
                    continue

                # Get similarity scores
                similarity_results = analyzer.analyze_response({
                    "question": question,
                    "options": {"A": correct_answer},
                    "base_output": sample.get('base_output', ''),
                    "prompting_output": sample.get('prompting_output', ''),
                    "cot_output": sample.get('cot_output', ''),
                    "medrag_output": sample.get('medrag_output', ''),
                    "internetsearch_output": sample.get('internetsearch_output', '')
                })

                # Process each method
                for method in ['base', 'prompting', 'cot', 'medrag', 'internetsearch']:
                    output_key = f"{method}_output"
                    if output_key not in sample:
                        continue

                    output = sample[output_key]
                    method_stats = model_performance[model_name][method]

                    if output is None:
                        continue

                    # Convert to string if it's a dictionary
                    if isinstance(output, dict):
                        output = json.dumps(output)

                    if isinstance(correct_answer, dict):
                        correct_answer = json.dumps(correct_answer)

                    # Special handling for medrag output format
                    if method == 'medrag':
                        if any(letter in output for letter in ['A', 'B', 'C', 'D', 'E', 'F']):
                            if ',' in output:
                                output = output.split(',')[0].strip()

                            try:
                                output = options[output]
                            except:
                                pass

                    # Accuracy scoring
                    if correct_answer.lower() in output.lower() or str(correct_idx) in output:
                        method_stats["correct_sum"] += 1
                        method_stats["pointwise_score"] += P_c
                    else:
                        method_stats["pointwise_score"] += P_w

                    # Store similarity scores
                    if output_key in similarity_results['similarities']:
                        scores = similarity_results['similarities'][output_key]
                        for score_type, score in scores.items():
                            method_stats["similarity_scores"][score_type].append(score)
                            method_stats["all_scores"][f"{task}_{score_type}"].append(score)

                    method_stats["total_count"] += 1


    # Print results
    print("\n" + "="*80)
    print("[EVALUATION RESULTS - Averaged Performance Across Tasks]")
    print("="*80 + "\n")

    for model_name, methods in model_performance.items():
        print(f"Model: {model_name}")
        print("-" * 80)

        for method, stats in methods.items():
            total_count = stats["total_count"]
            if total_count > 0:
                avg_accuracy = (stats["correct_sum"] / total_count) * 100
                avg_pointwise = stats["pointwise_score"] / total_count

                print(f"\n  Method: {method.upper()}")
                print(f"    Accuracy: {avg_accuracy:.2f}% ({stats['correct_sum']}/{total_count})")
                print(f"    Pointwise Score: {avg_pointwise:.2f}")

                # Print similarity scores
                if stats["similarity_scores"]:
                    print("    Similarity Scores:")

                    # Overall scores
                    for score_type, scores in stats["similarity_scores"].items():
                        if scores:
                            avg_score = np.mean(scores)
                            std_score = np.std(scores)
                            print(f"      Overall {score_type}: {avg_score:.3f} (±{std_score:.3f})")

                    # Per-task scores
                    print("\n    Per-Task Similarity Scores:")
                    for task in tasks:
                        task_has_scores = False
                        for score_type in ["answer_similarity", "question_similarity", "combined_score"]:
                            scores = stats["all_scores"].get(f"{task}_{score_type}", [])
                            if scores:
                                if not task_has_scores:
                                    print(f"      {task}:")
                                    task_has_scores = True
                                avg_score = np.mean(scores)
                                std_score = np.std(scores)
                                print(f"        {score_type}: {avg_score:.3f} (±{std_score:.3f})")

        print("\n" + "="*80 + "\n")

    return model_performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical hallucination mitigation experiment results")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory containing experiment result JSON files")
    parser.add_argument("--dataset_dir", type=str, default="./dataset",
                        help="Directory containing ground truth dataset CSV files")
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names to evaluate (e.g., gpt-4o gemini-2.0-flash)")
    parser.add_argument("--tasks", nargs="+", default=['FCT', 'fake', 'nota'],
                        help="List of tasks to evaluate (default: FCT fake nota)")

    args = parser.parse_args()

    print(f"Evaluating results from: {args.results_dir}")
    print(f"Using ground truth from: {args.dataset_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Tasks: {', '.join(args.tasks)}\n")

    results = evaluate_medical_responses(
        results_dir=args.results_dir,
        dataset_dir=args.dataset_dir,
        models=args.models,
        tasks=args.tasks
    )

    print("\n[INFO] Evaluation complete!")
