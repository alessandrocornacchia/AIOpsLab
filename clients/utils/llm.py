# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""An common abstraction for a cached LLM inference setup. Currently supports OpenAI's gpt-4-turbo."""

import os
from openai import OpenAI
from pathlib import Path
<<<<<<< HEAD
from typing import Optional, List, Dict
from dataclasses import dataclass

import together

from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


CACHE_DIR = Path("./cache_dir")
CACHE_PATH = CACHE_DIR / "cache.json"
MODEL = "gpt-4-turbo-2024-04-09"
#MODEL = "deepseek-ai/DeepSeek-V3"
#MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"

def format_prompt(history):
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
=======
import json

CACHE_DIR = Path("./cache_dir")
CACHE_PATH = CACHE_DIR / "cache.json"
>>>>>>> upstream/main


class Cache:
    """A simple cache implementation to store the results of the LLM inference."""

    def __init__(self) -> None:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                self.cache_dict = json.load(f)
        else:
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.cache_dict = {}

    @staticmethod
    def process_payload(payload):
        if isinstance(payload, (list, dict)):
            return json.dumps(payload)
        return payload

    def get_from_cache(self, payload):
        payload_cache = self.process_payload(payload)
        if payload_cache in self.cache_dict:
            return self.cache_dict[payload_cache]
        return None

    def add_to_cache(self, payload, output):
        payload_cache = self.process_payload(payload)
        self.cache_dict[payload_cache] = output

    def save_cache(self):
        with open(CACHE_PATH, "w") as f:
            json.dump(self.cache_dict, f, indent=4)


class GPT4Turbo:
    """Abstraction for OpenAI's GPT-4 Turbo model."""

    def __init__(self):
        self.cache = Cache()

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model="gpt-4-turbo-2024-04-09",
                max_tokens=1024,
                temperature=0.5,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                timeout=60,
                stop=[],
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response

class TogetherLLM:
    """Abstraction for a Together AI model."""

    def __init__(self, api_key: Optional[str] = None, config_file: Optional[str] = None, use_cache: bool = True):
        self.cache = Cache() if use_cache else None
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        if not self.api_key and config_file:
            config = load_together_config(config_file)
            self.api_key = config.get("api_key")
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in TOGETHER_API_KEY environment variable")
        
        together.api_key = self.api_key

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        formatted_prompt = format_prompt(payload)  # Convert to a single string
    
        if self.cache:
            cache_result = self.cache.get_from_cache(formatted_prompt)
        if cache_result is not None:
            return cache_result

        try:
            response = together.Completion.create(
                model=MODEL,
                prompt=formatted_prompt,  # ✅ Now a string, not a list
                max_tokens=1024,
                temperature=0.5,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        output = [choice.text for choice in response.choices]
        return output


    def run(self, payload: List[Dict[str, str]]) -> List[str]:
        response = self.inference(payload)
        if self.cache:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response
