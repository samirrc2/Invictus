"""
invictus.llm — Centralized LLM Gateway
========================================
Single source of truth for all LLM calls in the platform.

Provider priority:  Gemini (if key present)  →  OpenAI  →  raise
All agents import from here. Zero LLM logic in individual agents.

Usage:
    from invictus.llm import call_llm_json, call_llm_text, llm_available

    if llm_available():
        result = call_llm_json(prompt)      # returns parsed dict
        text   = call_llm_text(prompt)       # returns raw string

    # Or with raw string return:
        raw = call_llm_json_raw(prompt)      # returns JSON string
"""
import json
import logging
import time as _t
from typing import Any, Dict, Optional

from invictus.config import (
    OPENAI_API_KEY, GEMINI_API_KEY,
    LLM_MODEL, LLM_MODEL_OPENAI, LLM_MODEL_GEMINI,
    LLM_PROVIDER, LLM_TEMPERATURE, LLM_MAX_TOKENS,
)

_log = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def llm_available() -> bool:
    """Check if any LLM provider is configured."""
    return bool(GEMINI_API_KEY or OPENAI_API_KEY)


def get_provider() -> str:
    """Return the active provider name: 'gemini', 'openai', or 'none'."""
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return "none"


def call_llm_json(prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
    """
    Call LLM and return parsed JSON dict.
    Raises on failure — caller handles fallback.
    """
    raw = call_llm_json_raw(prompt, temperature=temperature)
    return json.loads(raw)


def call_llm_json_raw(prompt: str, temperature: float = 0.1) -> str:
    """
    Call LLM and return raw JSON string.
    Raises on failure — caller handles fallback.
    """
    if GEMINI_API_KEY:
        return _call_gemini(prompt, temperature=temperature, json_mode=True)
    if OPENAI_API_KEY:
        return _call_openai(prompt, temperature=temperature, json_mode=True)
    raise RuntimeError("No LLM API key configured (set GEMINI_API_KEY or OPENAI_API_KEY in .env)")


def call_llm_text(
    prompt: str,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """
    Call LLM and return plain text response.
    Raises on failure — caller handles fallback.
    """
    if GEMINI_API_KEY:
        return _call_gemini(prompt, temperature=temperature, json_mode=False,
                            max_tokens=max_tokens)
    if OPENAI_API_KEY:
        return _call_openai(prompt, temperature=temperature, json_mode=False,
                            max_tokens=max_tokens)
    raise RuntimeError("No LLM API key configured (set GEMINI_API_KEY or OPENAI_API_KEY in .env)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROVIDER IMPLEMENTATIONS — private
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_gemini(
    prompt: str,
    temperature: float = 0.1,
    json_mode: bool = False,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)

    gen_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if json_mode:
        gen_config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        LLM_MODEL_GEMINI,
        generation_config=genai.GenerationConfig(**gen_config),
    )
    response = model.generate_content(prompt)
    return response.text


def _call_openai(
    prompt: str,
    temperature: float = 0.1,
    json_mode: bool = False,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {
        "model": LLM_MODEL_OPENAI,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
