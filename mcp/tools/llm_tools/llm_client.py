"""Provider-agnostic LLM client.

Provider precedence (controlled by env vars):
1. GEMINI_API_KEY  -> Google Gemini
2. OPENAI_API_KEY  -> OpenAI
3. ANTHROPIC_API_KEY -> Claude
4. (none)          -> deterministic template-based mock that still produces valid JSON

The mock provider matters: it lets the entire pipeline run end-to-end
without any API key and is what the unit tests exercise.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from shared.utils.logging import get_logger

log = get_logger("llm_client")

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMClient:
    """Single entry point for text + structured generation."""

    def __init__(self, prefer: Optional[str] = None):
        self.prefer = prefer or os.getenv("LLM_PROVIDER")
        self.provider, self.model = self._select_provider()
        log.info("LLM provider=%s model=%s", self.provider, self.model)

    # ---- provider selection ---------------------------------------------

    def _select_provider(self) -> tuple[str, str]:
        if self.prefer == "mock":
            return "mock", "mock-1.0"
        if (self.prefer == "gemini" or not self.prefer) and os.getenv("GEMINI_API_KEY"):
            try:
                import google.generativeai  # noqa: F401
                return "gemini", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            except ImportError:
                pass
        if (self.prefer == "openai" or not self.prefer) and os.getenv("OPENAI_API_KEY"):
            try:
                import openai  # noqa: F401
                return "openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            except ImportError:
                pass
        if (self.prefer == "anthropic" or not self.prefer) and os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                return "anthropic", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            except ImportError:
                pass
        return "mock", "mock-1.0"

    # ---- text generation -------------------------------------------------

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7,
                 max_tokens: int = 2000) -> LLMResponse:
        if self.provider == "gemini":
            return self._generate_gemini(prompt, system, temperature, max_tokens)
        if self.provider == "openai":
            return self._generate_openai(prompt, system, temperature, max_tokens)
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, system, temperature, max_tokens)
        return self._generate_mock(prompt, system)

    def _generate_gemini(self, prompt, system, temperature, max_tokens) -> LLMResponse:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system or None,
        )
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return LLMResponse(text=resp.text, provider="gemini", model=self.model)

    def _generate_openai(self, prompt, system, temperature, max_tokens) -> LLMResponse:
        from openai import OpenAI
        client = OpenAI()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(text=resp.choices[0].message.content or "",
                           provider="openai", model=self.model)

    def _generate_anthropic(self, prompt, system, temperature, max_tokens) -> LLMResponse:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return LLMResponse(text=text, provider="anthropic", model=self.model)

    def _generate_mock(self, prompt: str, system: str) -> LLMResponse:
        # Mock returns are produced inline by callers via templates.
        # When generate() is called directly on the mock, return a trivial echo.
        return LLMResponse(
            text=f"[mock-llm] {prompt[:120]}",
            provider="mock",
            model="mock-1.0",
        )

    # ---- structured (JSON) generation -----------------------------------

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str = "",
        temperature: float = 0.5,
        max_tokens: int = 3000,
        max_retries: int = 2,
    ) -> T:
        """Returns a validated Pydantic instance.

        For the mock provider the caller is expected to handle generation
        themselves; this method will raise NotImplementedError for it.
        """
        if self.provider == "mock":
            raise RuntimeError("mock provider cannot produce structured output; "
                               "callers should use a template fallback")

        sys_msg = (
            (system + "\n\n" if system else "")
            + "You MUST respond with a single valid JSON object that matches the requested schema. "
              "Do not include markdown fences, comments, or any text outside the JSON."
        )
        last_err = ""
        text = ""
        for attempt in range(max_retries + 1):
            text = self.generate(prompt, sys_msg, temperature, max_tokens).text
            cleaned = _strip_code_fences(text)
            try:
                obj = json.loads(cleaned)
                return schema.model_validate(obj)
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                log.warning("structured parse attempt %d failed: %s", attempt + 1, last_err)
                prompt = (
                    f"{prompt}\n\nPrevious response was invalid: {last_err}\n"
                    "Return only valid JSON matching the schema."
                )
        raise ValueError(f"failed to parse structured response: {last_err}\n---\n{text[:500]}")


_DEFAULT_CLIENT: Optional[LLMClient] = None


def get_llm_client(prefer: Optional[str] = None, force_new: bool = False) -> LLMClient:
    global _DEFAULT_CLIENT
    if force_new or _DEFAULT_CLIENT is None or (prefer and prefer != _DEFAULT_CLIENT.prefer):
        _DEFAULT_CLIENT = LLMClient(prefer=prefer)
    return _DEFAULT_CLIENT


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = _FENCE.sub("", text).strip()
    # Find first { ... last } if extra prose leaked in.
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)
    return text
