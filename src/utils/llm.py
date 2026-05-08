"""
LLM utilities for general text generation (insights, explanations).
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

async def generate_insight(prompt: str, system_prompt: str = "You are a helpful financial assistant.") -> str | None:
    """
    Generate a short financial insight using the best available LLM (Groq > OpenAI).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=groq_key)
            response = await client.chat.completions.create(
                model=os.getenv("GROQ_INSIGHT_MODEL", "llama-3.1-8b-instant"), # Use smaller model for quick insights
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=256,
                timeout=5.0,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.warning("Groq insight generation failed: %s", exc)

    if openai_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=openai_key)
            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=256,
                timeout=5.0,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.warning("OpenAI insight generation failed: %s", exc)

    return None
