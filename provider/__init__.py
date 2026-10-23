"""Official Cursor SDK model-provider plugin for Hermes Agent."""
from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

from .hermes_cursor_provider.client import CURSOR_API_URL

_SEED_MODELS = ("auto", "composer-2.5")


def _cursor_error_verdict(
    error: Any,
    *,
    status_code: int | None = None,
    error_code: str = "",
    message: str = "",
    body: dict | None = None,
    model: str = "",
) -> dict[str, Any] | None:
    """Terminal verdict for the plugin's fail-closed plan-only raise, and nothing else.

    The profile hook runs for EVERY Cursor failure before the status and message paths,
    so an unscoped dict would make a 429 usage limit non-retryable and non-fallback —
    disabling credential rotation and provider fallback for all of them. Decline unless
    the structured code proves this is our own raise.
    """
    code = str(error_code or "").strip().lower()
    # Two different failures, two different reasons. A plan-only exhaustion is not a malformed
    # request: the model announced an action, was nudged twice, and never produced a call --
    # ``incomplete_response`` says exactly that, and ``format_error`` would tell the user to
    # start a clean session over a request that was fine. A rejected model parameter *is* a bad
    # request, so it keeps ``format_error`` and its accurate copy.
    reasons = {
        "cursor_plan_only": "incomplete_response",
        "cursor_invalid_params": "format_error",
    }
    reason = reasons.get(code)
    if reason is None:
        # A third raise site carries no code at all: ``_split_model`` rejects a malformed
        # ``id@key=value`` string -- a typo like ``grok-4.7@xhigh``, a repeated ``@``, an empty
        # id, a duplicate key -- before the bridge exists, with a plain ValueError whose text
        # names our own parser. The host reads a ValueError as a local validation bug and answers
        # the turn from the next provider, silently. Recognise our own text and opt out; the host
        # honours a verdict that does so even when the exception is a ValueError subclass.
        first_arg = getattr(error, "args", ())
        text = str(first_arg[0]) if first_arg else ""
        if text.startswith("Cursor model parameters"):
            return {"reason": "format_error", "retryable": False, "should_fallback": False}
        return None
    return {"reason": reason, "retryable": False, "should_fallback": False}


class CursorProfile(ProviderProfile):
    def create_client(self, **kwargs: Any):
        from .hermes_cursor_provider.client import CursorSDKClient

        return CursorSDKClient(**kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        # Never forward a Cursor key through a caller-supplied endpoint.
        if base_url and str(base_url).rstrip("/") != CURSOR_API_URL:
            return None
        try:
            from .hermes_cursor_provider.client import list_cursor_models

            models = list_cursor_models(api_key=api_key or "", timeout=timeout)
        except Exception:
            return None
        return models or None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        scope = str(context.get("cache_scope_id") or context.get("session_id") or "").strip()
        return ({}, {"_cursor_session_scope": scope}) if scope else ({}, {})


profile = CursorProfile(
    name="cursor",
    aliases=("cursor-sdk", "cursor-subscription"),
    display_name="Cursor SDK",
    description="Cursor subscription through the official Python SDK; Hermes owns tool execution.",
    signup_url="https://cursor.com/dashboard/api",
    api_mode="chat_completions",
    base_url=CURSOR_API_URL,
    hostname="api.cursor.com",
    auth_type="api_key",
    env_vars=("CURSOR_API_KEY",),
    supports_health_check=False,
    supports_model_listing=True,
    supports_vision=True,
    fallback_models=_SEED_MODELS,
    model_aliases={"default": "auto", "composer": "composer-2.5"},
    model_capabilities={
        model: {"supports_vision": True, "supports_tools": True}
        for model in _SEED_MODELS
    },
    default_aux_model="auto",
    unsupported_response_formats=("json_schema", "json_object"),
    # Constructor kwarg, never a method: the dataclass field default would shadow a
    # same-named method in the generated __init__.
    classify_api_error=_cursor_error_verdict,
)
register_provider(profile)
