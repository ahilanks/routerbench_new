import os
from together import Together
from openai import OpenAI
from xai_sdk import Client
from xai_sdk.chat import user, system
import tiktoken
import time
from dataclasses import dataclass
from google import genai


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
            # reasoning chunks (not user-visible)
            if event.type == "response.output_text.reasoning.delta":
                if event.delta:
                    reasoning += event.delta

            # visible output
            elif event.type == "response.output_text.delta":
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                answer += event.delta or ""

            # final event with token usage
            elif event.type == "response.completed":
                total_latency = time.perf_counter() - start_time
                if hasattr(event, "response") and hasattr(event.response, "usage"):
                    input_tokens = event.response.usage.input_tokens
                    output_tokens = event.response.usage.output_tokens

        # make sure the stream fully closes
        stream.close()

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
        self.client = Together(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )

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
            if hasattr(chunk, "choices") and chunk.choices:
                delta = getattr(chunk.choices[0].delta, "content", None)
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)

                # skip reasoning stream
                if reasoning:
                    continue

                if delta:
                    #print(delta, end="", flush=True)
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - start_time
                    answer += delta

            if getattr(chunk, "usage", None) is not None:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                total_latency = time.perf_counter() - start_time
                break

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

        chat = self.client.chat.create(model=model)
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
            # Handle reasoning content separately
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            delta = getattr(chunk.choices[0].delta, "content", None)

            # print(chunk)

            # Only count latency when first "content" (not reasoning) token arrives
            if delta is not None:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                answer += delta

            # Track usage if available (usually only in the final chunk)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                output_tokens = getattr(chunk.usage, "completion_tokens", 0)
                total_latency = time.perf_counter() - start_time


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
            reasoning_delta = ""

            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0].delta
                if hasattr(choice, "reasoning_content") and choice.reasoning_content:
                    reasoning_delta = choice.reasoning_content  # capture it
                if hasattr(choice, "content") and choice.content:
                    delta = choice.content

            # Append reasoning tokens to the answer (for full transcript)
            if reasoning_delta:
                answer += reasoning_delta

            # Append visible tokens, but use these to trigger latency
            if delta:
                #print(delta, end="", flush=True)
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                answer += delta

            # Handle usage info once stream completes
            if getattr(chunk, "usage", None) is not None:
                total_latency = time.perf_counter() - start_time
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                output_tokens = getattr(chunk.usage, "completion_tokens", 0)
                break

        # Fallback in case no usage is sent
        if total_latency is None:
            total_latency = time.perf_counter() - start_time

        result = {
            'answer': answer.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': first_token_latency,
            'total_latency': total_latency,
        }

        return result




class GoogleProvider:
    def __init__(self):
        self.client = genai.Client()

    def chat(self, model, system_prompt, query):
        answer = ""
        input_tokens = 0
        output_tokens = 0
        first_token_latency = None
        total_latency = None
        start_time = time.perf_counter()

        stream = self.client.models.generate_content_stream(
            model=model,
            contents=[system_prompt + "\n\n" + query]
        )

        for chunk in stream:
            # print(chunk)

            if chunk.candidates and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if getattr(part, "text", None):
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - start_time
                    answer += part.text

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.prompt_token_count or input_tokens
                output_tokens = chunk.usage_metadata.candidates_token_count or output_tokens

        total_latency = time.perf_counter() - start_time

        result = {
            "answer": answer.strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "first_token_latency": first_token_latency,
            "total_latency": total_latency,
        }

        return result




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


# response = GoogleProvider().chat('gemini-2.5-flash-lite', 'You are a helpful assistant.', 'What is the capital of France?')
# print(response)
