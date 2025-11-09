# from providers import OpenAIProvider, TogetherProvider, XAIProvider, DeepInfraProvider, HyperbolicProvider, FireworksProvider
# from datasets import load_dataset
# import pandas as pd
# import re
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# import json
# import numpy as np

# #config
# OUTPUT_PATH = "data/all_providers_results.jsonl"
# MAX_WORKERS = 24 
# RESUME = True
# SIZE = 22

# ds = load_dataset("cais/mmlu", "all")
# ds = ds["auxiliary_train"].to_pandas()
# ds['choices'] = ds['choices'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
# ds = ds.sample(frac=1, random_state=42).reset_index(drop=True)
# ds = ds.iloc[:SIZE]

# providers = {OpenAIProvider(): ['gpt-5-nano-2025-08-07'], 
#              TogetherProvider(): ['google/gemma-3n-E4B-it'], 
#              XAIProvider(): ['grok-4-fast-reasoning'], 
#              DeepInfraProvider(): ['openai/gpt-oss-120b',
#                                    'openai/gpt-oss-20b',
#                                    'nvidia/NVIDIA-Nemotron-Nano-9B-v2',
#                                    'meta-llama/Llama-3.2-11B-Vision-Instruct'], 
#             HyperbolicProvider(): ['Qwen/Qwen3-235B-A22B'],
#             FireworksProvider(): ['accounts/fireworks/models/kimi-k2-thinking']}


# prices = {
#     'gpt-5-nano-2025-08-07': (0.05, 0.40),
#     'google/gemma-3n-E4B-it': (0.02, 0.04),
#     'grok-4-fast-reasoning': (0.20, .40),
#     'openai/gpt-oss-120b': (0.05, 0.24),
#     'openai/gpt-oss-20b': (0.03, 0.14),
#     'nvidia/NVIDIA-Nemotron-Nano-9B-v2': (0.04, 0.16),
#     'meta-llama/Llama-3.2-11B-Vision-Instruct': (0.049, 0.049),
#     'Qwen/Qwen3-235B-A22B': (0.40, 0.40),
#     'accounts/fireworks/models/kimi-k2-thinking': (1.20, 1.20),
# }

# USER_PROMPT_TEMPLATE = """
# Here is a multiple-choice question. Read it carefully and answer it naturally, as you would in a conversation setting.

# Question:
# {QUESTIONS}

# Choices (each labeled with an index):
# 0: {CHOICES[0]}
# 1: {CHOICES[1]}
# 2: {CHOICES[2]}
# 3: {CHOICES[3]}

# Instructions:
# - Respond naturally in whatever length feels appropriate — short or detailed, depending on the question.
# - You MUST end your response with the exact phrase:
# Final Answer: <index_of_correct_choice>

# Format your response exactly like this example:
# Explanation/Reasoning: <your explanation/Reasoning>
# Final Answer: 2
# """


# # logic
# done_ids = set()
# if RESUME:
#     try:
#         with open(OUTPUT_PATH, 'r') as f:
#             for line in f:
#                 j = json.loads(line)
#                 done_ids.add(j['index'])
#     except FileNotFoundError:
#         pass

# # one worker
# def run_model(provider, model, question, choices):
#     user_prompt = USER_PROMPT_TEMPLATE.format(QUESTIONS=question, CHOICES=choices)
#     try:
#         return provider.chat(model, "You are a helpful assistant.", user_prompt)
#     except Exception as e:
#         return f"Error: {e}"

# # main loop
# start = time.time()

# with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool, open(OUTPUT_PATH, 'a') as fout:
#     for i in range(len(ds)):
#         if i in done_ids:
#             continue

#         q = ds.iloc[i]
#         futures = []
#         for provider, models in providers.items():
#             for model in models:
#                 futures.append((provider, model, pool.submit(run_model, provider, model, q['question'], q['choices'])))

#         row = {
#             'index': i,
#             'question': q['question'],
#             'choices': q['choices'],
#             'correct_answer': int(q['answer']),
#         }

#         for provider, model, future in futures:
#             row[model + '_response'] = future.result()

#         # atomic write per question
#         fout.write(json.dumps(row) + '\n')
#         fout.flush()  # ensure it’s persisted
#         if i % 50 == 0:
#             print(f"[{i}/{len(ds)}] written ({time.time()-start:.1f}s)")

# print(f"All done in {time.time()-start:.1f}s")
