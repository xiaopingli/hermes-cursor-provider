"""The plan-only Cursor failure is terminal: Hermes neither retries it nor falls back.

Runs under the Hermes-core interpreter (``~/.hermes/hermes-agent/venv``), which has both
pytest and an editable install of the core tree. The plugin's own venv cannot import
``agent.error_classifier``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Optional core dependency: skip cleanly under the plugin venv, where the module is
# genuinely absent, but stay loud under the core venv when an installed core is broken.
hermes_cli = pytest.importorskip(
    "hermes_cli",
    exc_type=ModuleNotFoundError,
    reason="needs the Hermes core editable install",
)

# ``providers`` imports ``hermes_cli.update_channel``, which imports the repo-local ``pm``
# package. The editable install does not expose ``pm``, so put the core root on sys.path
# rather than depending on the caller's working directory.
_CORE_ROOT = Path(hermes_cli.__file__).resolve().parents[1]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

import providers
from agent.error_classifier import FailoverReason, classify_api_error

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "provider"
_MODULE = "_cursor_provider_tree_under_test"
_NAMES = {"cursor", "cursor-sdk", "cursor-subscription"}


def _load_tree_provider():
    """Import the tree's plugin package by location (conftest only adds ``provider/``)."""
    module = sys.modules.get(_MODULE)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        _MODULE, PROVIDER / "__init__.py", submodule_search_locations=[str(PROVIDER)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree_profile(monkeypatch):
    module = _load_tree_provider()
    original = providers.get_provider_profile

    def resolve(name):
        if str(name).strip().lower() in _NAMES:
            return module.profile
        return original(name)

    # ``get_provider_profile`` prefers the bound home's plugin layer over ``_REGISTRY``,
    # and the installed cursor-provider (no hook) wins there; resolve the tree's profile.
    monkeypatch.setattr(providers, "get_provider_profile", resolve)
    assert callable(module.profile.classify_api_error)
    return module


def _error(module, message, **kwargs):
    client = sys.modules[f"{_MODULE}.hermes_cursor_provider.client"]
    return client.CursorSDKRunError(message, **kwargs)


def test_plan_only_error_is_terminal(tree_profile):
    exc = _error(
        tree_profile,
        "Cursor announced work twice without a tool call or a FINAL: answer",
        status_code=400,
        error_code="cursor_plan_only",
        body={"error": {"code": "cursor_plan_only"}},
    )
    verdict = classify_api_error(exc, provider="cursor", model="grok-4.7")
    # Not format_error: the request was well formed; the model simply never acted, and
    # format_error's terminal copy would misdiagnose it as a malformed request.
    assert verdict.reason == FailoverReason.incomplete_response
    assert verdict.retryable is False
    assert verdict.should_fallback is False


def test_invalid_parameters_error_is_terminal(tree_profile):
    """A typo'd model parameter is permanent: retrying it burns runs, and a fallback hides it.

    Live: ``reasoning_effort=bogus`` produced three failed runs and then a silent answer
    from ``xai-oauth/grok-4.7``. The code ``cursor_invalid_params`` must stop both.
    """
    exc = _error(
        tree_profile,
        'AI Model Not Found Invalid parameters for registry model: "grok-4.7"',
        status_code=400,
        error_code="cursor_invalid_params",
        body={"error": {"code": "cursor_invalid_params"}},
    )
    verdict = classify_api_error(exc, provider="cursor", model="grok-4.7")
    assert verdict.reason == FailoverReason.format_error
    assert verdict.retryable is False
    assert verdict.should_fallback is False


def test_usage_limit_keeps_its_own_verdict(tree_profile):
    """The hook is scoped: a 429 must keep rotating credentials / falling back."""
    exc = _error(tree_profile, "You're out of usage. Switch to Auto.")
    assert exc.status_code == 429
    verdict = classify_api_error(exc, provider="cursor", model="grok-4.7")
    assert verdict.reason == FailoverReason.rate_limit
    assert verdict.status_code == 429
    assert verdict.should_rotate_credential is True


def test_run_timeout_is_not_swallowed_by_the_plan_only_hook(tree_profile):
    """The hook stays scoped to ``cursor_plan_only``: a stalled run must stay retryable.

    A stall is transient, so it deliberately does NOT get the plan-only terminal
    verdict; it keeps the core's own 408/timeout handling (retry, not fail-closed).
    """
    exc = _error(
        tree_profile,
        "Cursor SDK run did not finish within 1800s (1800s elapsed); Hermes stopped waiting",
        status_code=408,
        error_code="cursor_run_timeout",
        body={"error": {"code": "cursor_run_timeout"}},
    )
    verdict = classify_api_error(exc, provider="cursor", model="grok-4.7")
    assert verdict.reason == FailoverReason.timeout
    assert verdict.retryable is True
    assert verdict.should_fallback is False


# A THIRD raise site carries no code at all: ``_split_model`` rejects a malformed
# ``id@key=value`` string before the bridge exists, with a plain ValueError. The host reads a
# ValueError as a local validation bug and answers the turn from another provider unless the
# profile hook recognises its own text and opts out. Pin every shape the parser can emit, and pin
# that an unrelated ValueError is still declined — a hook that swallowed all of them would turn
# genuine local bugs into terminal errors with no fallback at all.
@pytest.mark.parametrize(
    "value",
    [
        "grok-4.7@xhigh",            # a part that is not key=value
        "grok-4.7@reasoning_effort", # likewise, no '='
        "grok-4.7@a=1,a=2",          # a repeated key
        "@fast=true",                # an empty model id
        # NB the fork's parser accepts a second '@' (the model id keeps it), so that shape is not
        # a raise here; the upstream tag's parser rejects it. Only shapes that raise are pinned.
    ],
)
def test_a_malformed_parameter_string_is_terminal_not_a_local_bug(tree_profile, value):
    client = sys.modules[f"{_MODULE}.hermes_cursor_provider.client"]
    with pytest.raises(ValueError) as excinfo:
        client._split_model(value)
    verdict = classify_api_error(excinfo.value, provider="cursor", model="grok-4.7")
    assert verdict.reason == FailoverReason.format_error
    assert verdict.retryable is False
    assert verdict.should_fallback is False


def test_an_unrelated_value_error_is_still_declined(tree_profile):
    verdict = classify_api_error(ValueError("boom"), provider="cursor", model="grok-4.7")
    assert verdict.reason == FailoverReason.unknown
    assert verdict.should_fallback is False
