"""Shared connection settings for the Qwen model served by vLLM on the GB10."""

import os


BASE_URL = os.getenv("VLLM_BASE_URL", "http://10.184.47.255:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
REQUEST_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "90"))
