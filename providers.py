import os
from together import Together
from openai import OpenAI
from xai_sdk import Client
from xai_sdk.chat import user, system
import tiktoken
import time
from dataclasses import dataclass


class OpenAIProvider:
    def __init__(self):
        self.client = OpenAI()

    def chat(self, model, system_prompt, query):
        answer = ""
        first_token_latency = None
        total_latency = None
        input_tokens = 0
        output_tokens = 0
        start_time = time.perf_counter()

        stream = self.client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            stream=True,
        )

        for event in stream:
            if event.type == "response.output_text.delta":
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                answer_chunk = event.delta or ""
                #print(answer_chunk, end="", flush=True)
                answer += answer_chunk
            elif event.type == "response.completed":
                total_latency = time.perf_counter() - start_time
                input_tokens = event.response.usage.input_tokens
                output_tokens = event.response.usage.output_tokens

        result = {
            'answer': answer.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': first_token_latency,
            'total_latency': total_latency,
        }

        return result


class TogetherProvider:
    def __init__(self):
        self.client = Together()

    def chat(self, model, system_prompt, query):
        answer = ""
        input_tokens = 0
        output_tokens = 0
        first_token_latency = None
        total_latency = None
        start_time = time.perf_counter()

        stream = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            stream=True,
        )


        for chunk in stream:

            if chunk.choices and chunk.choices[0]:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                answer_chunk = chunk.choices[0].delta.content or ""
                answer += answer_chunk
                #print(answer_chunk, end="", flush=True)
            else:
                total_latency = time.perf_counter() - start_time
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        result = {
            'answer': answer.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': first_token_latency,
            'total_latency': total_latency,
        }

        return result


class XAIProvider:
    def __init__(self):
        self.client = Client(
            api_key=os.getenv("XAI_API_KEY")
        )

        
    def chat(self, model, system_prompt, query):
        answer = ""
        input_tokens = 0
        output_tokens = 0
        first_token_latency = None
        total_latency = None
        start_time = time.perf_counter()

        chat = self.client.chat.create(model="grok-4")
        chat.append(system(system_prompt))
        chat.append(user(query))
        for response, chunk in chat.stream():
            if first_token_latency is None:
                first_token_latency = time.perf_counter() - start_time
            #print(chunk.content, end="", flush=True)
        
        total_latency = time.perf_counter() - start_time
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens + response.usage.reasoning_tokens
        answer = response.content

        result = {
            'answer': answer.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': first_token_latency,
            'total_latency': total_latency,
        }

        return result


class DeepInfraProvider:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def chat(self, model, system_prompt, query):
        answer = ""
        input_tokens = 0
        output_tokens = 0
        first_token_latency = None
        total_latency = None
        start_time = time.perf_counter()

        stream = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content != None:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                answer_chunk = chunk.choices[0].delta.content or ""
                answer += answer_chunk
                #print(answer_chunk, end="", flush=True)
            else:
                total_latency = time.perf_counter() - start_time
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        result = {
            'answer': answer.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': first_token_latency,
            'total_latency': total_latency,
        }

        return result


class HyperbolicProvider:


    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("HYPERBOLIC_API_KEY"),
            base_url="https://api.hyperbolic.xyz/v1",
        )

    def chat(self, model, system_prompt, query):
        answer = ""
        input_tokens = 0
        output_tokens = 0
        in_think = True
        first_token_latency = None
        total_latency = None
        start_time = time.perf_counter()

        stream = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            stream=True,
        )

        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                total_latency = time.perf_counter() - start_time
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                break

            delta = chunk.choices[0].delta.content or ""
            answer += delta

            if "</think>" in delta:
                in_think = False
                post_think = delta.split("</think>", 1)[1]
                if post_think.strip():
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - start_time
                    #print(post_think, end="", flush=True)
                continue

            if "<think>" in delta:
                in_think = True
                continue
            if not in_think:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                #print(delta, end="", flush=True)

        result = {
            'answer': answer.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': first_token_latency,
            'total_latency': total_latency,
        }

        return result




class FireworksProvider:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1",
        )

    def chat(self, model, system_prompt, query):
        answer = ""
        input_tokens = 0
        output_tokens = 0
        in_think = True
        first_token_latency = None
        total_latency = None
        start_time = time.perf_counter()

        stream = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            stream=True,
        )

        for chunk in stream:
            delta = ""
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0].delta
                if hasattr(choice, "reasoning_content") and choice.reasoning_content:
                    continue
                if hasattr(choice, "content") and choice.content:
                    delta = choice.content

            if delta:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                answer += delta
                #print(delta, end="", flush=True)

            # Handle usage when stream ends
            if getattr(chunk, "usage", None) is not None:
                total_latency = time.perf_counter() - start_time
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                break

        result = {
            'answer': answer.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': first_token_latency,
            'total_latency': total_latency,
        }

        return result