from providers import OpenAIProvider, TogetherProvider, XAIProvider, DeepInfraProvider, HyperbolicProvider, FireworksProvider
from datasets import load_dataset
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import numpy as np

#config
OUTPUT_PATH = "data/all_providers_results.jsonl"
MAX_WORKERS = 6   # modest concurrency per provider
RESUME = True
SIZE = 10

ds = load_dataset("cais/mmlu", "all")
ds = ds["auxiliary_train"].to_pandas()
ds['choices'] = ds['choices'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
ds = ds.sample(frac=1, random_state=42).reset_index(drop=True)
ds = ds.iloc[:SIZE]

providers = {OpenAIProvider(): ['gpt-5-nano-2025-08-07'], 
             TogetherProvider(): ['google/gemma-3n-E4B-it'], 
             XAIProvider(): ['grok-4-fast-reasoning'], 
             DeepInfraProvider(): ['openai/gpt-oss-120b',
                                   'openai/gpt-oss-20b',
                                   'nvidia/NVIDIA-Nemotron-Nano-9B-v2',
                                   'meta-llama/Llama-3.2-11B-Vision-Instruct'], 
            HyperbolicProvider(): ['Qwen/Qwen3-235B-A22B'],
            FireworksProvider(): ['accounts/fireworks/models/kimi-k2-thinking']}


prices = {
    'gpt-5-nano-2025-08-07': (0.05, 0.40),
    'google/gemma-3n-E4B-it': (0.02, 0.04),
    'grok-4-fast-reasoning': (0.20, .40),
    'openai/gpt-oss-120b': (0.05, 0.24),
    'openai/gpt-oss-20b': (0.03, 0.14),
    'nvidia/NVIDIA-Nemotron-Nano-9B-v2': (0.04, 0.16),
    'meta-llama/Llama-3.2-11B-Vision-Instruct': (0.049, 0.049),
    'Qwen/Qwen3-235B-A22B': (0.40, 0.40),
    'accounts/fireworks/models/kimi-k2-thinking': (1.20, 1.20),
}

USER_PROMPT_TEMPLATE = """
Here is a multiple-choice question. Read it carefully and answer it naturally, as you would in a conversation setting.

Question:
{QUESTIONS}

Choices (each labeled with an index):
0: {CHOICES[0]}
1: {CHOICES[1]}
2: {CHOICES[2]}
3: {CHOICES[3]}

Instructions:
- Respond naturally in whatever length feels appropriate — short or detailed, depending on the question.
- You MUST end your response with the exact phrase:
Final Answer: <index_of_correct_choice>

Format your response exactly like this example:
Explanation/Reasoning: <your explanation/Reasoning>
Final Answer: 2
"""


def run_one(i):
    user_prompt = USER_PROMPT_TEMPLATE.format(QUESTIONS=ds.iloc[i]['question'], CHOICES=ds.iloc[i]['choices'])
    response = OpenAIProvider().chat("gpt-5-nano-2025-08-07", "You are a helpful assistant.", user_prompt)

    return {
        'question': ds.iloc[i]['question'],
        'choices': ds.iloc[i]['choices'],
        'correct_answer': int(ds.iloc[i]['answer']),

        'gpt-5-nano-2025-08-07_response': response,
    }



start_time = time.time()
results = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(run_one, i): i for i in range(SIZE)}
    for future in as_completed(futures):
        try:
            results.append(future.result())
        except Exception as e:
            print(f"Error on index {futures[future]}: {e}")

print(f"Completed {len(results)} in {time.time() - start_time:.2f}s")

# Write out JSONL
with open('data/openai-gpt-5-nano-2025-08-07.jsonl', 'w') as f:
    for row in results:
        f.write(json.dumps(row) + '\n')