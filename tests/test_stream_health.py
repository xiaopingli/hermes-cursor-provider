"""Tests for scripts/stream_health.py — the summariser that reads a Hermes log.

The tool's job is attribution, so its tests are about honesty as much as parsing: a close line
with no session tag must be reported as inferred, and an unexplained window must be reported as
unexplained rather than summarised into false confidence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "stream_health.py"
_spec = importlib.util.spec_from_file_location("stream_health", SCRIPT)
stream_health = importlib.util.module_from_spec(_spec)
sys.modules["stream_health"] = stream_health
_spec.loader.exec_module(stream_health)


HEAD = """2026-10-19 08:15:52,139 INFO [20261019_074356_1a423c4e] agent.conversation_loop: API call #1: model=grok-4.7@context=500k provider=cursor in=1241717 out=10 total=1241727 latency=106.9s cache=1/2 (0%) id=x
2026-10-19 08:17:41,825 WARNING [20261019_074356_1a423c4e] agent.chat_completion_helpers: Interrupted provider wait counted as stale after 44s with no output; consecutive stale attempts=1.
2026-10-19 08:17:57,853 INFO run_agent: OpenAI client closed (stream_error_cleanup, shared=False, tcp_force_closed=0) thread=Thread-65 (<lambda>):1 provider=cursor base_url=https://api.cursor.com model=grok-4.7@context=500k
"""


def _write(tmp_path, body: str) -> Path:
    path = tmp_path / "agent.log"
    path.write_text(body, encoding="utf-8")
    return path


def test_collects_calls_stales_and_closes(tmp_path):
    report = stream_health.collect(_write(tmp_path, HEAD), None, None)
    assert len(report.calls) == 1
    assert report.calls[0].in_tokens == 1_241_717
    assert report.calls[0].session == "20261019_074356_1a423c4e"
    assert [s.seconds for s in report.stales] == [44]
    assert len(report.closes) == 1


def test_close_inherits_the_preceding_session_and_is_marked_inferred(tmp_path):
    """The core's close line has no session id: the tag must be flagged, never asserted."""
    report = stream_health.collect(_write(tmp_path, HEAD), None, None)
    assert report.closes[0].session == "20261019_074356_1a423c4e"
    assert report.closes[0].inferred is True
    text = stream_health.render(report, tmp_path / "agent.log", "", None, None)
    assert "(inferred)" in text


def test_close_with_no_preceding_session_is_unattributed(tmp_path):
    body = """2026-10-19 08:17:57,853 INFO run_agent: OpenAI client closed (stream_error_cleanup, shared=False, tcp_force_closed=0) thread=Thread-65 provider=cursor base_url=https://api.cursor.com
"""
    report = stream_health.collect(_write(tmp_path, body), None, None)
    assert report.closes[0].inferred is False
    assert "unattributed" in stream_health.render(report, tmp_path / "agent.log", "", None, None)


def test_window_filters_out_other_minutes(tmp_path):
    body = HEAD + "2026-10-19 09:30:00,000 INFO [20261019_074356_1a423c4e] agent.conversation_loop: API call #2: model=grok-4.7 provider=cursor in=5 out=1 total=6 latency=1.0s\n"
    path = _write(tmp_path, body)
    day = stream_health.peek_day(path)
    since = stream_health.parse_time("08:15", day=day)
    until = stream_health.parse_time("08:20", day=day)
    report = stream_health.collect(path, since, until)
    # The 08:15:52 call is in the window; the 09:30 one is not.
    assert [c.in_tokens for c in report.calls] == [1_241_717]
    assert [s.seconds for s in report.stales] == [44]
    assert len(report.closes) == 1


def test_health_line_is_parsed_and_reported(tmp_path):
    body = """2026-10-19 21:00:00,000 INFO [20261019_074356_1a423c4e] hermes_cursor_provider.client: cursor: stream health mode=persistent close=timeout events=0 ttfe=-1.00s gap=0.00s dur=1800.00s in_tokens=0 prompt_chars=12345 session=20261019_074356_1a423c4e
"""
    report = stream_health.collect(_write(tmp_path, body), None, None)
    assert len(report.runs) == 1
    run = report.runs[0]
    assert (run.mode, run.close, run.events, run.in_tokens, run.prompt_chars) == (
        "persistent",
        "timeout",
        0,
        0,
        12345,
    )
    text = stream_health.render(report, tmp_path / "agent.log", "", None, None)
    assert "no-event" in text          # the ttfe sentinel is rendered, not hidden as -1.0
    assert "produced no event at all" in text


def test_a_window_with_no_health_lines_says_so(tmp_path):
    """The 08:17-08:29 revision emitted none; the summary must not read as 'all clean'."""
    report = stream_health.collect(_write(tmp_path, HEAD), None, None)
    text = stream_health.render(report, tmp_path / "agent.log", "", None, None)
    assert "the plugin revision there did not emit them" in text


def test_session_filter_keeps_only_that_session(tmp_path):
    body = HEAD + """2026-10-19 08:26:04,821 WARNING [20261019_081441_fa808499] agent.chat_completion_helpers: Interrupted provider wait counted as stale after 54s with no output; consecutive stale attempts=1.
"""
    path = _write(tmp_path, body)
    report = stream_health.collect(path, None, None)
    report.stales = [s for s in report.stales if s.session == "20261019_081441_fa808499"]
    assert [s.seconds for s in report.stales] == [54]


def test_parse_time_rejects_nonsense():
    with pytest.raises(SystemExit):
        stream_health.parse_time("half past nine")


def test_parse_time_needs_a_day_for_a_bare_clock():
    with pytest.raises(SystemExit):
        stream_health.parse_time("08:17")
