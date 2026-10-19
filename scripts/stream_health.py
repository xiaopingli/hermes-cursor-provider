#!/usr/bin/env python3
"""Summarise Cursor-provider stream health from a Hermes log file.

Answers one question: what happened inside each Cursor run — how long before the first
event, the longest silence, how it closed, and how many tokens it re-read — instead of
leaving a stall to be guessed at from a wall-clock gap between unrelated lines.

    python3 scripts/stream_health.py LOGFILE --session 20261019_074356_1a423c4e
    python3 scripts/stream_health.py LOGFILE --since 08:17 --until 08:29

Reads four line shapes, all of which the log already carries once the plugin emits
``cursor: stream health``:

* ``cursor: stream health ...``          — the plugin's own per-run line (session stamped)
* ``API call #N: ... provider=cursor``   — the core's per-call accounting (session stamped)
* ``stream_error_cleanup ... provider=cursor`` — the core closing a client (NO session tag)
* ``counted as stale after Ns``          — the core abandoning a provider wait (session stamped)

Two honesty rules, because a summary that invents attribution is worse than none:

1. The core's close lines carry no session id. When one is attributed, it is attributed from
   the nearest *preceding* stamped line and printed as ``inferred``. Two sessions can
   interleave in one process, so an inferred tag is a hypothesis, not a fact.
2. Events the log cannot explain are reported as unattributed rather than dropped. A close
   with no matching health line is exactly the shape this tool exists to expose.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

STAMP = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
SESSION_TAG = re.compile(r"^[^]]*\[(?P<sid>\d{8}_\d{6}_[0-9a-f]+)\]")
HEALTH = re.compile(
    r"cursor: stream health mode=(?P<mode>\S+) close=(?P<close>\S+) events=(?P<events>\d+) "
    r"ttfe=(?P<ttfe>-?[\d.]+)s gap=(?P<gap>[\d.]+)s dur=(?P<dur>[\d.]+)s "
    r"in_tokens=(?P<in_tokens>\d+) prompt_chars=(?P<prompt_chars>\d+) session=(?P<session>\S+)"
)
API_CALL = re.compile(
    r"API call #(?P<n>\d+): .*provider=cursor in=(?P<in_tokens>\d+) out=(?P<out>\d+) "
    r"total=(?P<total>\d+) latency=(?P<latency>[\d.]+)s"
)
CLOSE = re.compile(r"OpenAI client closed \(stream_error_cleanup.*provider=cursor")
STALE = re.compile(r"counted as stale after (?P<secs>\d+)s with no output; consecutive stale attempts=(?P<n>\d+)")


@dataclass
class Run:
    ts: datetime
    session: str
    mode: str
    close: str
    events: int
    ttfe: float
    gap: float
    dur: float
    in_tokens: int
    prompt_chars: int


@dataclass
class Call:
    ts: datetime
    session: str
    number: int
    in_tokens: int
    out_tokens: int
    latency: float


@dataclass
class Close:
    ts: datetime
    session: str = ""
    inferred: bool = False


@dataclass
class Stale:
    ts: datetime
    session: str
    seconds: int
    consecutive: int


@dataclass
class Report:
    runs: list[Run] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    closes: list[Close] = field(default_factory=list)
    stales: list[Stale] = field(default_factory=list)
    scanned: int = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None


def parse_time(text: str, *, day: datetime | None = None) -> datetime:
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    if day is not None:
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(text, fmt).replace(year=day.year, month=day.month, day=day.day)
            except ValueError:
                continue
    raise SystemExit(f"stream_health: cannot read the time {text!r}")


def collect(path: Path, since: datetime | None, until: datetime | None) -> Report:
    report = Report()
    last_session = ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stamp = STAMP.match(raw)
            if not stamp:
                continue
            ts = datetime.strptime(stamp.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            if report.first_ts is None:
                report.first_ts = ts
            report.last_ts = ts
            if since is not None and ts < since:
                continue
            if until is not None and ts > until:
                continue
            report.scanned += 1
            tagged = SESSION_TAG.match(raw[24:]) or SESSION_TAG.match(raw)
            if tagged:
                last_session = tagged.group("sid")

            health = HEALTH.search(raw)
            if health:
                report.runs.append(
                    Run(
                        ts=ts,
                        session=health.group("session"),
                        mode=health.group("mode"),
                        close=health.group("close"),
                        events=int(health.group("events")),
                        ttfe=float(health.group("ttfe")),
                        gap=float(health.group("gap")),
                        dur=float(health.group("dur")),
                        in_tokens=int(health.group("in_tokens")),
                        prompt_chars=int(health.group("prompt_chars")),
                    )
                )
                continue

            call = API_CALL.search(raw)
            if call:
                report.calls.append(
                    Call(
                        ts=ts,
                        session=last_session,
                        number=int(call.group("n")),
                        in_tokens=int(call.group("in_tokens")),
                        out_tokens=int(call.group("out")),
                        latency=float(call.group("latency")),
                    )
                )
                continue

            if CLOSE.search(raw):
                # The core's close line has no session tag: inherit the nearest preceding
                # stamped one and say so. Never present this as observed fact.
                report.closes.append(Close(ts=ts, session=last_session, inferred=bool(last_session)))
                continue

            stale = STALE.search(raw)
            if stale:
                report.stales.append(
                    Stale(
                        ts=ts,
                        session=last_session,
                        seconds=int(stale.group("secs")),
                        consecutive=int(stale.group("n")),
                    )
                )
    return report


def fmt_ts(ts: datetime) -> str:
    return ts.strftime("%H:%M:%S")


def render(report: Report, path: Path, session: str, since: datetime | None, until: datetime | None) -> str:
    out: list[str] = []
    span = "n/a"
    if report.first_ts and report.last_ts:
        span = f"{report.first_ts:%Y-%m-%d %H:%M:%S} .. {report.last_ts:%H:%M:%S}"
    window = []
    if since:
        window.append(f"since {fmt_ts(since)}")
    if until:
        window.append(f"until {fmt_ts(until)}")
    if session:
        window.append(f"session {session}")
    out.append(f"file    {path}")
    out.append(f"span    {span}   ({report.scanned} line(s) in window{' — ' + ', '.join(window) if window else ''})")
    out.append("")

    out.append(f"runs    {len(report.runs)} plugin health line(s)")
    if report.runs:
        out.append("        time     session               mode        close             ttfe     gap      dur      in_tokens  prompt")
        for run in report.runs:
            ttfe = "no-event" if run.ttfe < 0 else f"{run.ttfe:.1f}s"
            out.append(
                f"        {fmt_ts(run.ts):<8} {run.session:<21} {run.mode:<11} {run.close:<17} "
                f"{ttfe:<8} {run.gap:>6.1f}s {run.dur:>7.1f}s {run.in_tokens:>10}  {run.prompt_chars}"
            )
    out.append("")

    out.append(f"calls   {len(report.calls)} cursor API call(s)")
    if report.calls:
        biggest = max(report.calls, key=lambda c: c.in_tokens)
        slowest = max(report.calls, key=lambda c: c.latency)
        out.append(f"        largest input   {biggest.in_tokens:,} tokens at {fmt_ts(biggest.ts)} (call #{biggest.number})")
        out.append(f"        slowest         {slowest.latency:.1f}s at {fmt_ts(slowest.ts)} (call #{slowest.number})")
        over = [c for c in report.calls if c.in_tokens >= 500_000]
        if over:
            out.append(f"        over 500k       {len(over)}: " + ", ".join(f"{c.in_tokens:,}" for c in over))
    out.append("")

    out.append(f"closes  {len(report.closes)} stream_error_cleanup line(s)")
    for close in report.closes:
        tag = f"{close.session} (inferred)" if close.inferred else "unattributed"
        out.append(f"        {fmt_ts(close.ts)}  {tag}")
    out.append("")

    out.append(f"stale   {len(report.stales)} abandoned provider wait(s)")
    for stale in report.stales:
        label = stale.session or "unattributed"
        out.append(
            f"        {fmt_ts(stale.ts)}  {label}  no output for {stale.seconds}s (consecutive {stale.consecutive})"
        )
    out.append("")

    findings: list[str] = []
    if report.closes:
        findings.append(
            f"{len(report.closes)} close(s) with no session tag in the core's own line — attributed here only by "
            "preceding-line inference"
        )
    silent = [r for r in report.runs if r.ttfe < 0]
    if silent:
        findings.append(f"{len(silent)} run(s) produced no event at all before closing (ttfe sentinel)")
    wide = [r for r in report.runs if r.gap >= 30.0]
    if wide:
        worst = max(wide, key=lambda r: r.gap)
        findings.append(f"{len(wide)} run(s) had a silent gap >= 30s (worst {worst.gap:.1f}s at {fmt_ts(worst.ts)})")
    if not report.runs:
        findings.append("no plugin health lines in this window — the plugin revision there did not emit them")
    out.append("findings")
    for item in findings or ["none"]:
        out.append(f"  - {item}")
    return "\n".join(out)


def peek_day(path: Path) -> datetime | None:
    """The date of the file's first timestamp, so ``--since 08:17`` needs no date."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stamp = STAMP.match(raw)
            if stamp:
                return datetime.strptime(stamp.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarise Cursor-provider stream health from a Hermes log.")
    parser.add_argument("logfile", type=Path, help="the log file to read (agent.log*, .gz not supported)")
    parser.add_argument("--session", default="", help="only lines for this session id")
    parser.add_argument("--since", default="", help="HH:MM[:SS] or a full timestamp")
    parser.add_argument("--until", default="", help="HH:MM[:SS] or a full timestamp")
    args = parser.parse_args(argv)

    if not args.logfile.is_file():
        raise SystemExit(f"stream_health: no such file: {args.logfile}")

    day = peek_day(args.logfile)
    since = parse_time(args.since, day=day) if args.since else None
    until = parse_time(args.until, day=day) if args.until else None
    if since and until and until < since:
        until += timedelta(days=1)

    report = collect(args.logfile, since, until)
    if args.session:
        report.runs = [r for r in report.runs if r.session == args.session]
        report.calls = [c for c in report.calls if c.session == args.session]
        report.stales = [s for s in report.stales if s.session == args.session]
        # Closes carry no session of their own; keep the inferred ones and label them.
        report.closes = [c for c in report.closes if c.session == args.session]

    print(render(report, args.logfile, args.session, since, until))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
