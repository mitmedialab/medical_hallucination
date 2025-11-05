"""
Medical Hallucination Mitigation Experiments

This script runs experiments across multiple LLM models using different hallucination mitigation methods.
It supports OpenAI, Google Gemini, DeepSeek, and HuggingFace-based medical models.

Prerequisites:
1. Clone MedRAG repository
2. Set up API keys in .env file
3. Download required datasets
4. Set up MedRAG corpus (optional, for retrieval-augmented generation)

Usage:
    python run_experiments.py --models gpt-4o gemini-2.0-flash --seed 0 --output_dir ./results

For more details, see README.md
"""

import os
import time
from dotenv import load_dotenv
from itertools import zip_longest
import json
from tqdm import tqdm
import gc
import pandas as pd
import argparse

from google import genai
import openai
from openai import OpenAI
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Import MedRAG from the cloned repository
import sys
sys.path.insert(0, './MedRAG/src')
from medrag import MedRAG

import random
import numpy as np

# Import Tavily for internet search
from tavily import TavilyClient


def try_three_times(func, *args, **kwargs):
    """Retry function up to 3 times with exponential backoff"""
    for attempt in range(3):
        try:
            result = func(*args, **kwargs)
            if result is not None:
                return result
            print(f"Attempt {attempt + 1} returned None for {func.__name__}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
            if "rate limit" in str(e).lower():
                # Exponential backoff for rate limits
                wait_time = 2 ** (attempt + 1)
                print(f"Rate limit detected. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                time.sleep(1)
    print(f"{func.__name__} failed after 3 attempts.")
    return None


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Load environment variables from .env file
load_dotenv()

# Load API keys from environment variables (REQUIRED)
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize Tavily client if API key is available
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


def convert_to_format(input_text):
    """Convert input text to question and options format"""
    lines = input_text.strip().split('\n')
    question = lines[0]

    options_lines = lines[2:]  # Skip the first two lines ("Options:" part)
    options = {}

    # Convert numbers to letters (1->A, 2->B, etc.)
    number_to_letter = {str(i): chr(64 + i) for i in range(1, 27)}  # 1->A, 2->B, ..., 26->Z

    for line in options_lines:
        if '.' in line:
            key, value = line.split('.', 1)
            key = key.strip()
            if key in number_to_letter:
                key = number_to_letter[key]
            options[key] = value.strip()

    return {
        "question": question,
        "options": options
    }


def setup_inference_optimizations():
    """Setup CUDA optimizations for better inference performance"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True


def clear_gpu_memory():
    """Clear GPU memory and cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class MedicalModelInference:
    def __init__(self, model_name):
        """Initialize the medical model inference class"""
        self.model_name = model_name
        self.system_prompt = "You are a truthful medical assistant. Please provide a detailed and concise response without any hallucinations."
        self.cot_instruction = "Let's think step-by-step."

        # Determine model class based on model name patterns
        model_lower = model_name.lower()

        # OpenAI models (GPT, O1, O3 series)
        if any(pattern in model_lower for pattern in ['gpt', 'o1', 'o3', 'o4']):
            self.model_type = model_name
            self.model_class = "openai"
        # Google Gemini models (excluding MedGemma which is HuggingFace)
        elif 'gemini' in model_lower and 'gemma' not in model_lower:
            self.model_type = model_name
            self.model_class = "gemini"
        # MedGemma - Special handling for chat model
        elif 'medgemma' in model_lower:
            self.model_type = model_name
            self.model_class = "medgemma"
        # Regular Gemma models
        elif 'gemma' in model_lower:
            self.model_type = model_name
            self.model_class = "google"
        # DeepSeek models
        elif 'deepseek' in model_lower:
            self.model_type = model_name
            self.model_class = "deepseek"
        # AlpaCare models
        elif "alpacare" in model_lower:
            self.model_type = model_name
            self.model_class = "xz97"
        # MedAlpaca models
        elif "medalpaca" in model_lower:
            self.model_type = model_name
            self.model_class = "medalpaca"
        # Meditron models
        elif "meditron" in model_lower:
            self.model_type = model_name
            self.model_class = "epfl-llm"
        # PMC-LLaMA models
        elif "pmc" in model_lower:
            self.model_type = model_name
            self.model_class = "axiong"
        # Default to transformer for unknown models
        else:
            self.model_type = model_name
            self.model_class = "transformer"


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.client = None # Initialize client to None
        self.pipeline = None # Initialize pipeline for MedGemma

        # Initialize API clients for cloud-based models
        if self.model_class == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
            self.client = OpenAI(api_key=openai_api_key)
        elif self.model_class == "gemini":
            if not google_api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file")
            self.client = genai.Client(api_key=google_api_key)
        elif self.model_class == "deepseek":
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in your .env file")
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        # Note: MedGemma and other HuggingFace models don't need API clients


    def load_model(self):
        print(f"Loading {self.model_type}...")

        offload_folder = f"./offloaded_weights_{self.model_type}"
        os.makedirs(offload_folder, exist_ok=True)
        cache_dir="./model_cache"
        os.makedirs(cache_dir, exist_ok=True)

        # Special handling for MedGemma - use pipeline API as recommended
        if self.model_class == "medgemma":
            try:
                from transformers import pipeline
                print("Loading MedGemma using pipeline API...")

                # Create pipeline for MedGemma
                self.pipeline = pipeline(
                    "image-text-to-text",
                    model="google/medgemma-4b-it",
                    torch_dtype=torch.bfloat16,
                    device=self.device,
                    cache_dir=cache_dir
                )

                # For compatibility, set tokenizer from pipeline
                self.tokenizer = self.pipeline.tokenizer
                self.model = self.pipeline.model  # For cleanup purposes

                print("Successfully loaded MedGemma pipeline")
                return True
            except Exception as e:
                print(f"Error loading MedGemma: {e}")
                import traceback
                traceback.print_exc()
                return False

        # Build model path for other models
        elif self.model_class == "google":
            model_path = f"google/{self.model_name}"
        else:
            model_path = f"{self.model_class}/{self.model_name}"

        # Special handling for Meditron models
        if "meditron" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=True
            )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config,
                local_files_only=True,
                max_memory={i: "24GiB" for i in range(4)},
            )
        # MedGemma and other smaller models with 4-bit quantization
        elif "gemma" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir
            )
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quantization_config,
                offload_folder=offload_folder
            )
        # Standard loading for other models
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                offload_folder=offload_folder
            )

        self.model.eval()

        print(f"Successfully loaded {self.model_type}")
        return True

    def run_base(self, text):
        """Run basic inference without additional prompting"""
        if self.model_class == "openai":
            try:
                # GPT-5 and O-series models have different parameter requirements
                params = {
                    "model": self.model_type,
                    "messages": [{"role": "user", "content": text}]
                }
                if "gpt-5" in self.model_type.lower() or "o1" in self.model_type.lower() or "o3" in self.model_type.lower():
                    # GPT-5/O-series: use max_completion_tokens, temperature=1 (default, don't specify)
                    params["max_completion_tokens"] = 1024
                else:
                    # GPT-4 and earlier: use max_tokens and custom temperature
                    params["max_tokens"] = 1024
                    params["temperature"] = 0.7

                completion = self.client.chat.completions.create(**params)
                return completion.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API error: {e}")
                raise

        elif self.model_class == "gemini":
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=text
                )
                return response.text
            except Exception as e:
                print(f"Gemini API error: {e}")
                raise

        elif self.model_class == "deepseek":
            try:
                completion = self.client.chat.completions.create(
                    model="deepseek/deepseek-r1",
                    messages=[
                        {
                            "role": "user",
                            "content": text
                        }
                    ],
                    temperature=0.7,
                    max_completion_tokens=1024
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"DeepSeek API error: {e}")
                raise

        elif self.model_class == "medgemma":
            # Handle MedGemma with proper chat template
            return self._run_medgemma(text, method="base")
        else:
            return self.gen_response(text)

    def run_prompting(self, text):
        """Run inference with system prompt"""
        prompting_input = f"{self.system_prompt}\n{text}"

        if self.model_class == "openai":
            # GPT-5 and O-series models have different parameter requirements
            params = {
                "model": self.model_type,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]
            }
            if "gpt-5" in self.model_type.lower() or "o1" in self.model_type.lower() or "o3" in self.model_type.lower():
                # GPT-5/O-series: use max_completion_tokens, temperature=1 (default, don't specify)
                params["max_completion_tokens"] = 1024
            else:
                # GPT-4 and earlier: use max_tokens and custom temperature
                params["max_tokens"] = 1024
                params["temperature"] = 0.7

            completion = self.client.chat.completions.create(**params)
            return completion.choices[0].message.content

        elif self.model_class == "gemini":
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompting_input
            )
            return response.text

        elif self.model_class == 'deepseek':
            completion = self.client.chat.completions.create(
                model="deepseek/deepseek-r1",
                messages=[
                    {
                    "role": "user",
                    "content": prompting_input
                    }
                ]
            )
            return completion.choices[0].message.content

        elif self.model_class == "medgemma":
            # Handle MedGemma with system prompt
            return self._run_medgemma(text, method="prompting")
        else:
            return self.gen_response(prompting_input)

    def run_cot(self, text):
        """Run inference with chain of thought prompting"""
        cot_input = f"{self.system_prompt}\n{self.cot_instruction}\n{text}"

        if self.model_class == "openai":
            # GPT-5 and O-series models have different parameter requirements
            params = {
                "model": self.model_type,
                "messages": [
                    {"role": "system", "content": self.system_prompt + "\n" + self.cot_instruction},
                    {"role": "user", "content": text}
                ]
            }
            if "gpt-5" in self.model_type.lower() or "o1" in self.model_type.lower() or "o3" in self.model_type.lower():
                # GPT-5/O-series: use max_completion_tokens, temperature=1 (default, don't specify)
                params["max_completion_tokens"] = 1024
            else:
                # GPT-4 and earlier: use max_tokens and custom temperature
                params["max_tokens"] = 1024
                params["temperature"] = 0.7

            completion = self.client.chat.completions.create(**params)
            return completion.choices[0].message.content

        elif self.model_class == "gemini":
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=cot_input
            )
            return response.text

        elif self.model_class == "deepseek":
            completion = self.client.chat.completions.create(
                model="deepseek/deepseek-r1",
                messages=[
                    {
                    "role": "user",
                    "content": cot_input
                    }
                ]
            )
            return completion.choices[0].message.content

        elif self.model_class == "medgemma":
            # Handle MedGemma with CoT
            return self._run_medgemma(text, method="cot")
        else:
            return self.gen_response(cot_input)

    def run_medrag(self, question, options, k=32, corpus_dir="./MedRAG/corpus"):
        """Run inference with MedRAG"""
        # For MedGemma, we need to use a custom approach
        # MedGemma uses "image-text-to-text" pipeline which is incompatible with MedRAG's "text-generation"
        if self.model_class == "medgemma":
            try:
                sys.path.insert(0, './MedRAG/src')
                from utils import RetrievalSystem

                # Use only the retrieval system since MedRAG can't handle MedGemma's pipeline type
                retrieval_system = RetrievalSystem(
                    retriever_name="MedCPT",
                    corpus_name="Textbooks",
                    db_dir=corpus_dir,
                    cache=True
                )

                # Get retrieved documents
                retrieved_snippets, scores = retrieval_system.retrieve(question, k=k)

                # Format context from retrieved snippets (use top 5 most relevant)
                contexts = []
                for idx, snippet in enumerate(retrieved_snippets[:5]):
                    # Extract title and content from snippet
                    title = snippet.get('title', 'Unknown')
                    content = snippet.get('content', '')
                    contexts.append(f"Reference [{idx+1}] (Title: {title})\n{content}")

                context = "\n\n".join(contexts)

                # Format options
                options_str = '\n'.join([f"{key}. {value}" for key, value in sorted(options.items())])

                # Create RAG-enhanced prompt similar to MedRAG's format
                rag_prompt = f"""You are a medical expert. Use the following references to answer the question.

References:
{context}

Question: {question}

Options:
{options_str}

Based on the references above, select the most appropriate answer. Respond with only the letter (A, B, C, or D) of the correct option."""

                # Use MedGemma to generate answer
                answer = self._run_medgemma(rag_prompt, method="base")
                print(f"4. medrag: {answer}")
                print()

                if answer:
                    # Extract answer choice from response
                    import re
                    # Look for pattern like "A", "B", "C", "D" in the answer
                    # First try to find standalone letter
                    match = re.search(r'^([A-D])[\.:\s)]?', answer.strip())
                    if match:
                        return match.group(1)

                    # Then try to find letter anywhere in the text
                    match = re.search(r'\b([A-D])\b', answer.upper())
                    if match:
                        return match.group(1)

                    # If no clear match, return None to indicate failure
                    print(f"Warning: Could not extract answer choice from MedGemma response: {answer}")
                    return None
                return None

            except Exception as e:
                print(f"MedRAG error for MedGemma: {e}")
                import traceback
                traceback.print_exc()
                return None

        # For other models, use standard MedRAG
        model_lower = self.model_name.lower()

        if self.model_class == "openai":
            model_prefix = "OpenAI"
        elif self.model_class == "gemini":
            model_prefix = "Google"
        elif self.model_class == "deepseek":
            model_prefix = "deepseek-ai"
            self.model_class = 'DeepSeek-R1'
        elif self.model_class == "google":  # Other Google models
            model_prefix = "google"
        elif 'alpacare' in model_lower:
            model_prefix = "xz97"
        elif 'medalpaca' in model_lower:
            model_prefix = "medalpaca"
        elif 'pmc' in model_lower:
            model_prefix = "axiong"
        elif 'meditron' in model_lower:
            model_prefix = "epfl-llm"
        else:
            # Default to using the model class directly
            model_prefix = self.model_class

        medrag = MedRAG(
            llm_name=f"{model_prefix}/{self.model_name}",
            rag=True,
            retriever_name="MedCPT",
            corpus_name="Textbooks",
            corpus_cache=True
        )
        try:
            answer, snippets, scores = medrag.answer(question=question, options=options, k=k)
            print("4. medrag:", answer)
            print()

            # MedRAG returns a JSON string with step_by_step_thinking and answer_choice
            if answer:
                try:
                    # Clean up any markdown code blocks
                    cleaned_string = answer.replace('```json', '').replace('```', '').strip()
                    dictionary = json.loads(cleaned_string)
                    # Extract the answer_choice field
                    answer_choice = dictionary.get('answer_choice', None)
                    if answer_choice:
                        return answer_choice
                    else:
                        # If no answer_choice, try to extract from the response
                        import re
                        match = re.search(r'\b([A-D])\b', answer.upper())
                        if match:
                            return match.group(1)
                        return answer
                except json.JSONDecodeError:
                    # If not JSON, try to extract answer directly
                    import re
                    match = re.search(r'\b([A-D])\b', answer.upper())
                    if match:
                        return match.group(1)
                    return answer
            return None
        except Exception as e:
            print(f"MedRAG error: {e}")
            return None

        time.sleep(1)

    def run_internet_search_agent(self, user_query):
        """
        Pipeline: Input --> Search (Tavily) --> Input to LLM --> Answer
        """
        if not tavily_client:
            print("Warning: Tavily client not initialized. Skipping internet search.")
            return "Internet search not available - Tavily API key missing"

        try:
            # Step 1. Executing a simple search query
            res = tavily_client.search(user_query[:400])

            search_result = ""
            for idx,result in enumerate(res["results"]):
                search_result += f"[{idx+1}]\nTitle: {result['title']}\nContent: {result['content']}\n\n"


            # Step 2: Combine search result with the user's query
            combined_input = (
                f"User Query: {user_query}\n"
                f"Search Result: {search_result}\n"
                "Based on the information above, provide a comprehensive answer."
            )

            # Step 3: Generate the final answer using the LLM
            if self.model_class == "openai":
                completion = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=[{"role": "user", "content": combined_input}]
                )
                return completion.choices[0].message.content

            elif self.model_class == "gemini":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=combined_input
                )
                return response.text

            elif self.model_class == "deepseek":
                completion = self.client.chat.completions.create(
                    model="deepseek/deepseek-r1",
                    messages=[
                        {
                            "role": "user",
                            "content": combined_input
                        }
                    ]
                )
                return completion.choices[0].message.content

            elif self.model_class == "medgemma":
                # Handle MedGemma for internet search
                return self._run_medgemma(combined_input, method="base")
            else:
                # Other HuggingFace models
                return self.gen_response(combined_input)

        except Exception as e:
            print(f"Internet search error: {e}")
            return f"Error during internet search: {str(e)}"

    def _run_medgemma(self, text, method="base"):
        """Run MedGemma using pipeline API for text-only tasks"""
        if not self.pipeline:
            print("MedGemma pipeline not loaded")
            return None

        try:
            # Build the prompt based on method
            if method == "prompting":
                system_content = self.system_prompt
                user_content = text
            elif method == "cot":
                system_content = f"{self.system_prompt}\n{self.cot_instruction}"
                user_content = text
            else:
                system_content = None
                user_content = text

            # Format messages for the pipeline (text-only)
            if system_content:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_content}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_content}]
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_content}]
                    }
                ]

            # Generate using pipeline
            output = self.pipeline(
                text=messages,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

            # Extract the generated text from pipeline output
            if output and len(output) > 0:
                if isinstance(output[0], dict) and "generated_text" in output[0]:
                    generated = output[0]["generated_text"]
                    # The response is typically in the last message
                    if isinstance(generated, list) and len(generated) > 0:
                        last_message = generated[-1]
                        if isinstance(last_message, dict) and "content" in last_message:
                            return last_message["content"]
                    return str(generated)
                else:
                    return str(output[0])

            return None

        except Exception as e:
            print(f"MedGemma inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def gen_response(self, prompts, max_new_tokens=1024, batch_size=1):
        """Generate response for transformer models"""
        if not isinstance(prompts, list):
            prompts = [prompts]

        responses = []

        for i in range(0, len(prompts), batch_size):
            clear_gpu_memory()
            batch = prompts[i:i + batch_size]

            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=64,
                    padding=True
                ).to(self.model.device)

                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        top_p=0.95,
                        top_k=50,
                        repetition_penalty=1.1
                    )

                batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_responses = [
                    response[len(prompt):] if response.startswith(prompt) else response
                    for prompt, response in zip(batch, batch_responses)
                ]
                responses.extend([r.strip() for r in batch_responses])

                del outputs, inputs
                clear_gpu_memory()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    clear_gpu_memory()
                    print("OOM error, trying with reduced memory usage...")
                    if batch_size > 1:
                        return self.gen_response(prompts, max_new_tokens, batch_size=1)
                    else:
                        print("Error even with batch size 1, skipping this batch")
                        responses.extend(["Error: Out of memory"] * len(batch))
                else:
                    raise e

        return responses[0] if len(prompts) == 1 else responses

    def cleanup(self):
        """Clean up GPU memory"""
        # Clean up for all HuggingFace-based models
        if self.model_class not in ["openai", "gemini", "deepseek"]:
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            if hasattr(self, 'pipeline') and self.pipeline:
                del self.pipeline
                self.pipeline = None
            if hasattr(self, 'processor') and self.processor:
                del self.processor
                self.processor = None
            clear_gpu_memory()


def create_result_data(questions, options, base_results, prompting_results, cot_results, medrag_results, internetsearch_results):
    """Create a structured result dictionary"""
    return [
        {
            "question": question,
            "options": option,
            "base_output": base,
            "prompting_output": prompting,
            "cot_output": cot,
            "medrag_output": medrag,
            "internetsearch_output": internetsearch
        }
        for question, option, base, prompting, cot, medrag, internetsearch in zip_longest(
            questions, options, base_results, prompting_results, cot_results, medrag_results, internetsearch_results,
            fillvalue=None
        )
    ]


def save_results(result_data, model_name, file_name, output_path, seed=None, intermediate=False):
    """Save results to a JSON file"""
    suffix = "_intermediate" if intermediate else ""
    result_file = os.path.join(output_path, f"{model_name}_{file_name.split('.')[0]}{suffix}_seed{seed}.json")
    result_dict = {
        "seed": seed,
        "results": result_data
    }
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)


def main(args):
    dataset_path = args.dataset_path
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    setup_inference_optimizations()

    model_names = args.models

    for model_name in model_names:
        print(f"\n[INFO] Processing model: {model_name}")
        medical_model = MedicalModelInference(model_name)

        # Load model for HuggingFace-based models
        if medical_model.model_class not in ["openai", "gemini", "deepseek"]:
            try:
                if not medical_model.load_model():
                    print(f"Skipping {model_name} due to loading error")
                    continue
                else:
                    print(f"Model loaded: {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                continue

        for file_name in os.listdir(dataset_path):
            if file_name in ['medhalt_reasoning_fake.csv', 'medhalt_reasoning_nota.csv', 'medhalt_reasoning_FCT.csv']:
                result_file = os.path.join(output_path, f"{model_name}_{file_name.split('.')[0]}_seed{args.seed}.json")
                if os.path.exists(result_file):
                    print(f"Result file already exists: {result_file}. Skipping.")
                    continue

                print("Processing file:", file_name)
                file_path = os.path.join(dataset_path, file_name)
                data = pd.read_csv(file_path)

                if "prompt" not in data.columns:
                    print(f"Skipping {file_name}: 'prompt' column not found.")
                    continue

                base_results = []
                prompting_results = []
                cot_results = []
                questions = []
                options = []
                medrag_results = []
                internetsearch_results = []

                for count, prompt in enumerate(tqdm(data['prompt']), 1):
                    print(f"\n[INFO] Processing sample {count}/{len(data)}")

                    try:
                        output = convert_to_format(prompt)
                        question = output['question']
                        option = output['options']

                        questions.append(question)
                        options.append(option)

                        # Run inference methods with retry logic
                        base_result = try_three_times(medical_model.run_base, prompt)
                        base_results.append(base_result)
                        print(f'1. base: {base_result if base_result else "None"}')

                        prompting_result = try_three_times(medical_model.run_prompting, prompt)
                        prompting_results.append(prompting_result)
                        print(f'2. prompting: {prompting_result if prompting_result else "None"}')

                        cot_result = try_three_times(medical_model.run_cot, prompt)
                        cot_results.append(cot_result)
                        print(f'3. cot: {cot_result if cot_result else "None"}')

                    except Exception as e:
                        print(f"Error processing sample {count}: {e}")
                        # Append None for failed samples
                        if len(base_results) < count:
                            base_results.append(None)
                        if len(prompting_results) < count:
                            prompting_results.append(None)
                        if len(cot_results) < count:
                            cot_results.append(None)

                    # MedRAG
                    medrag_result = try_three_times(medical_model.run_medrag, question, option, k=32)
                    medrag_results.append(medrag_result)
                    print(f"4. medrag: {medrag_result if medrag_result else 'None'}")

                    # Internet Search
                    internet_result = try_three_times(medical_model.run_internet_search_agent, prompt)
                    internetsearch_results.append(internet_result)
                    print(f"5. internet-search: {internet_result if internet_result else 'None'}")

                    # Save intermediate results every 10 samples
                    if count % 10 == 0:
                        result_data = create_result_data(
                            questions, options, base_results,
                            prompting_results, cot_results, medrag_results, internetsearch_results
                        )
                        save_results(
                            result_data, model_name, file_name, output_path, seed=args.seed, intermediate=True
                        )

                    time.sleep(1)

                # Save final results for this file
                result_data = create_result_data(
                    questions, options, base_results,
                    prompting_results, cot_results, medrag_results, internetsearch_results
                )

                save_results(
                    result_data, model_name, file_name, output_path, seed=args.seed, intermediate=False
                )

        # Cleanup after processing all files
        medical_model.cleanup()
        print(f"\n[INFO] Completed processing for {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run medical hallucination mitigation experiments")
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names to test (e.g., gpt-4o gemini-2.0-flash)")
    parser.add_argument("--dataset_path", type=str, default="./dataset",
                        help="Path to dataset directory containing CSV files")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save experiment results")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Using seed: {args.seed}")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    main(args)
