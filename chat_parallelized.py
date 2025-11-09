from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, numpy as np
from datasets import load_dataset
from providers import (
    OpenAIProvider, TogetherProvider, XAIProvider,
    DeepInfraProvider, HyperbolicProvider, FireworksProvider, GoogleProvider
)
import logging
import sys, os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/run.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
for noisy in [
    "httpx",
    "urllib3",
    "openai",
    "openai.afc",
    "openai._afc",
    "AFC",
    "requests",
]:
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger().addFilter(
    lambda record: "AFC" not in record.getMessage()
)




OUTPUT_PATH = "data/all_providers_results_parallelized.jsonl"
MAX_WORKERS = 256
RESUME = True
SIZE = 50000

# dataset
ds = load_dataset("cais/mmlu", "all")["auxiliary_train"].to_pandas()
ds['choices'] = ds['choices'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
ds = ds.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:SIZE]

providers = {
    OpenAIProvider(): ['gpt-5-nano-2025-08-07'],
    TogetherProvider(): ['google/gemma-3n-E4B-it'],
    XAIProvider(): ['grok-4-fast-reasoning'],
    DeepInfraProvider(): [
        'openai/gpt-oss-120b',
        'openai/gpt-oss-20b',
        'nvidia/NVIDIA-Nemotron-Nano-9B-v2',
        'meta-llama/Llama-3.2-11B-Vision-Instruct'
    ],
    #HyperbolicProvider(): ['Qwen/Qwen3-235B-A22B'],
    FireworksProvider(): ['accounts/fireworks/models/kimi-k2-thinking'],
    GoogleProvider(): ['gemini-2.5-flash-lite']
}

prices = {
    'gpt-5-nano-2025-08-07': (0.05, 0.40),
    'google/gemma-3n-E4B-it': (0.02, 0.04),
    'grok-4-fast-reasoning': (0.20, 0.50),
    'openai/gpt-oss-120b': (0.05, 0.24),
    'openai/gpt-oss-20b': (0.03, 0.14),
    'nvidia/NVIDIA-Nemotron-Nano-9B-v2': (0.04, 0.16),
    'meta-llama/Llama-3.2-11B-Vision-Instruct': (0.049, 0.049),
    #'Qwen/Qwen3-235B-A22B': (0.40, 0.40),
    'accounts/fireworks/models/kimi-k2-thinking': (1.20, 1.20),
    'gemini-2.5-flash-lite': (0.1, 0.4),
}

USER_PROMPT_TEMPLATE = """Here is a multiple-choice question. Read it carefully and answer it naturally, as you would in a conversation setting.

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

# resume from previous runs
done_ids = set()
if RESUME:
    try:
        with open(OUTPUT_PATH, 'r') as f:
            for line in f:
                j = json.loads(line)
                done_ids.add(j['index'])
    except FileNotFoundError:
        pass

# one worker
def run_task(provider, model, idx, q):
    prompt = USER_PROMPT_TEMPLATE.format(QUESTIONS=q['question'], CHOICES=q['choices'])
    try:
        r = provider.chat(model, "You are a helpful assistant.", prompt)
    except Exception as e:
        r = f"Error: {e}"
        logging.error(f"Error: {e} with model {model} and index {idx}")
    return idx, model, r

# all jobs
tasks = []
for i, row in ds.iterrows():
    if i in done_ids:
        continue
    for provider, models in providers.items():
        for model in models:
            tasks.append((provider, model, i, row))

start = time.time()
results_buffer = {}

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool, open(OUTPUT_PATH, 'a') as fout:
    futures = [pool.submit(run_task, *t) for t in tasks]
    for n, fut in enumerate(as_completed(futures), 1):
        idx, model, response = fut.result()
        model_key = model.replace('/', '_')

        # collect partial results
        if idx not in results_buffer:
            q = ds.iloc[idx]
            results_buffer[idx] = {
                'index': idx,
                'question': q['question'],
                'choices': q['choices'],
                'correct_answer': int(q['answer'])
            }
        results_buffer[idx][f'{model_key}_response'] = response

        # flush every N completions
        if n % 50 == 0:
            for row in results_buffer.values():
                fout.write(json.dumps(row) + '\n')
            fout.flush()
            results_buffer.clear()
            msg = f"[{n}/{len(tasks)}] done ({time.time()-start:.1f}s)"
            logging.info(msg)

# flush final leftovers
with open(OUTPUT_PATH, 'a') as fout:
    for row in results_buffer.values():
        fout.write(json.dumps(row) + '\n')
msg = f"Done in {time.time()-start:.1f}s"
logging.info(msg)
