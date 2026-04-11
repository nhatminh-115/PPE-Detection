"""
Phase 2 — LangGraph Multi-Agent Report Pipeline

Architecture:
  data_agent → pattern_agent → reliability_agent → report_agent → END

Each agent adds its results to the shared ReportState. The report_agent
produces the final narrative and persists it to daily_reports.

To generate a report manually:
    from src.api.report_pipeline import run_report_pipeline
    result = await run_report_pipeline()

This module is additive — the existing report_service.py is not modified.
Feature-flag the swap via REPORT_USE_LANGGRAPH=1 in .env when ready.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

ICT = timezone(timedelta(hours=7))


# ── Shared state ──────────────────────────────────────────────────────────────

class ReportState(TypedDict):
    report_date: str                  # YYYY-MM-DD in ICT
    # Data agent outputs
    violations: list[dict]            # raw rows from ppe_violations
    summary: dict                     # aggregate counts, peak_hour, etc.
    # Pattern agent outputs
    recurring_trackers: list[dict]    # trackers with >= 3 events
    systemic_flags: list[str]         # human-readable systemic risk notes
    # Reliability agent outputs
    quality: dict                     # from get_daily_quality
    label_coverage_note: str          # "adequate" | "sparse" | "no_data"
    # Report agent outputs
    narrative: str                    # final LLM narrative
    regulation_context: str           # RAG chunks injected into LLM prompt


# ── Data Agent ────────────────────────────────────────────────────────────────

def data_agent(state: ReportState) -> ReportState:
    """
    Fetch today's violation events from Supabase and compute aggregate statistics.
    Groups events by location (session_id), tracker, and hour.
    """
    from src.infrastructure.supabase import get_supabase_service_client

    report_date = state["report_date"]
    day_start = f"{report_date}T00:00:00+07:00"
    day_end   = f"{report_date}T23:59:59+07:00"

    client = get_supabase_service_client()
    rows: list[dict] = []
    if client:
        try:
            res = (
                client.table("ppe_violations")
                .select("*")
                .gte("created_at", day_start)
                .lt("created_at", day_end)
                .execute()
            )
            rows = res.data or []
        except Exception as exc:
            logger.warning("data_agent: Supabase query failed: %s", exc)

    # Aggregate
    violation_counts: dict[str, int] = defaultdict(int)
    tracker_counts:   dict[str, int] = defaultdict(int)
    hourly_counts:    dict[int, int]  = defaultdict(int)
    confidence_sum = 0.0
    confidence_n   = 0

    for row in rows:
        try:
            dt = datetime.fromisoformat(row["created_at"]).astimezone(ICT)
            hourly_counts[dt.hour] += 1
        except Exception:
            pass
        session_id = str(row.get("session_id") or "?")
        for v in row.get("violations", []):
            violation_counts[v.get("violation_type", "unknown")] += 1
            # Key by session_id:tracker_id — tracker IDs reset each session so
            # the same numeric ID across sessions means different people.
            tracker_key = f"{session_id}:{v.get('tracker_id', '?')}"
            tracker_counts[tracker_key] += 1
            conf = v.get("confidence")
            if conf is not None:
                confidence_sum += float(conf)
                confidence_n   += 1

    peak_hour    = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
    top_offender = max(tracker_counts, key=tracker_counts.get) if tracker_counts else None
    avg_conf     = round(confidence_sum / confidence_n, 4) if confidence_n else 0.0

    summary = {
        "total_events":       len(rows),
        "none_hardhat":       violation_counts.get("none_hardhat", 0),
        "none_vest":          violation_counts.get("none_vest", 0),
        "total_violations":   sum(violation_counts.values()),
        "peak_hour":          peak_hour,
        "top_offender_id":    top_offender,
        "top_offender_count": tracker_counts.get(top_offender, 0) if top_offender else 0,
        "avg_confidence":     avg_conf,
        "hourly_breakdown":   dict(hourly_counts),
        "tracker_breakdown":  dict(tracker_counts),
        "violation_counts":   dict(violation_counts),
    }

    logger.info(
        "data_agent: %d events, %d violations (hardhat=%d, vest=%d)",
        len(rows), summary["total_violations"],
        summary["none_hardhat"], summary["none_vest"],
    )

    return {**state, "violations": rows, "summary": summary}


# ── Pattern Agent ─────────────────────────────────────────────────────────────

def pattern_agent(state: ReportState) -> ReportState:
    """
    Detect recurring violations and systemic failure patterns.

    Flags:
    - Trackers with >= REPEAT_THRESHOLD violation events (unresolved loop).
    - High concentration in a single hour (> 50% of daily total).
    - Low average confidence (<0.5) suggests hardware/positioning issue.
    - Night-shift concentration (peak_hour 20-05) indicates coverage gap.
    """
    REPEAT_THRESHOLD = 3
    summary = state.get("summary", {})
    tracker_breakdown: dict[str, int] = summary.get("tracker_breakdown", {})
    hourly_breakdown:  dict[int, int]  = {int(k): v for k, v in summary.get("hourly_breakdown", {}).items()}
    total_events = summary.get("total_events", 0)
    peak_hour    = summary.get("peak_hour")
    avg_conf     = summary.get("avg_confidence", 1.0)

    # tracker_breakdown keys are "session_id:tracker_id" — display as short labels
    def _short(key: str) -> str:
        parts = key.split(":", 1)
        sess = parts[0][:8] if parts else key
        tid  = parts[1] if len(parts) > 1 else "?"
        return f"tracker {tid} (session {sess})"

    recurring = [
        {"tracker_key": tid, "event_count": cnt}
        for tid, cnt in tracker_breakdown.items()
        if cnt >= REPEAT_THRESHOLD
    ]
    recurring.sort(key=lambda x: x["event_count"], reverse=True)

    flags: list[str] = []

    if recurring:
        labels = ", ".join(_short(r["tracker_key"]) for r in recurring[:5])
        flags.append(
            f"{len(recurring)} tracker(s) have >= {REPEAT_THRESHOLD} repeated violations "
            f"within their session ({labels}). Systemic non-compliance suspected."
        )

    if total_events > 0 and peak_hour is not None:
        peak_share = hourly_breakdown.get(peak_hour, 0) / total_events
        if peak_share > 0.50:
            flags.append(
                f"Over 50% of violations concentrated in hour {peak_hour:02d}:00 ICT "
                f"({peak_share:.0%}). Suggests supervision gap during that period."
            )

    if total_events > 0 and avg_conf < 0.5:
        flags.append(
            f"Average detection confidence is low ({avg_conf:.2f}). "
            "Camera angle, lighting, or obstruction issues likely — prioritize hardware audit."
        )

    if peak_hour is not None and (peak_hour >= 20 or peak_hour <= 5):
        flags.append(
            f"Peak violations at {peak_hour:02d}:00 ICT (night shift). "
            "Reduced supervision and visibility — mandate PPE check-in protocol for night crew."
        )

    logger.info(
        "pattern_agent: %d recurring session-trackers, %d systemic flags",
        len(recurring), len(flags),
    )

    return {**state, "recurring_trackers": recurring, "systemic_flags": flags}


# ── Reliability Agent ─────────────────────────────────────────────────────────

def reliability_agent(state: ReportState) -> ReportState:
    """
    Pull today's human-label quality metrics from Phase 1 (ppe_labels).
    Attaches precision/recall estimates and coverage note to the state.
    """
    from src.api.label_service import get_daily_quality

    try:
        quality = get_daily_quality(state["report_date"])
    except Exception as exc:
        logger.warning("reliability_agent: get_daily_quality failed: %s", exc)
        quality = {
            "date": state["report_date"],
            "labeled_events": 0, "skipped": 0,
            "item_tp": 0, "item_fp": 0, "item_fn": 0, "item_tn": 0,
            "precision": None, "recall": None,
            "label_coverage_note": "no_data",
        }

    logger.info(
        "reliability_agent: %d labeled, precision=%s, recall=%s",
        quality.get("labeled_events", 0),
        quality.get("precision"), quality.get("recall"),
    )

    return {**state, "quality": quality, "label_coverage_note": quality.get("label_coverage_note", "no_data")}


# ── Report Agent ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a workplace safety analyst in Vietnam.
Given a daily PPE violation summary, pattern analysis, label-based quality metrics,
and relevant Vietnamese labor regulations, write a concise safety report in 4 short paragraphs:
1. Overview of the day's violations (counts, severity, shift context).
2. Systemic risk findings from pattern analysis (recurring offenders, concentrated hours).
3. Detection quality estimate — include precision/recall if labeled data is available;
   note confidence level if label coverage is sparse or absent.
4. Recommended corrective actions with mandatory regulation citation.

CITATION RULE: In paragraph 4, cite using:
(Thong tu 25/2022/TT-BLDTBXH, Dieu X) or (Thong tu 25/2022/TT-BLDTBXH, Phu luc I trang X)
Use real numbers from the regulation context only. Never write "Dieu X" as a placeholder.

ACTIONABILITY RULE: Recommendations must be specific to the data patterns:
- Repeat offender (>= 3 events): immediate suspension + retraining.
- Low confidence (<0.5): camera audit before worker training.
- Night shift peak: mandatory PPE check-in for night crew.
- Sparse labels (< 10): state that quality estimate is provisional.

Be direct and professional. Do not use bullet points. Keep under 300 words."""


def report_agent(state: ReportState) -> ReportState:
    """
    Fetch RAG regulation context, then generate narrative via Groq.
    Persists report to Supabase daily_reports table.
    """
    summary   = state.get("summary", {})
    quality   = state.get("quality", {})
    flags     = state.get("systemic_flags", [])
    report_date = state["report_date"]

    # Short-circuit: no activity today — skip LLM call entirely
    if summary.get("total_events", 0) == 0:
        narrative = f"[Hardcoded] No PPE violations recorded on {report_date}. No corrective action required."
        try:
            from src.infrastructure.supabase import get_supabase_service_client
            client = get_supabase_service_client()
            if client:
                client.table("daily_reports").upsert(
                    {
                        "report_date":    report_date,
                        "summary":        summary,
                        "narrative":      narrative,
                        "quality":        quality,
                        "systemic_flags": flags,
                        "engine":         "hardcoded",
                    },
                    on_conflict="report_date",
                ).execute()
                logger.info("report_agent: no activity on %s — hardcoded narrative persisted", report_date)
        except Exception as exc:
            logger.warning("report_agent: Supabase persist failed: %s", exc)
        return {**state, "narrative": narrative, "regulation_context": ""}

    # RAG context
    regulation_context = ""
    violation_types = list(summary.get("violation_counts", {}).keys())
    try:
        from rag.query import retrieve
        _VIOLATION_QUERIES = {
            "none_hardhat": "quy định bắt buộc đội mũ an toàn công nghiệp công nhân xây dựng",
            "none_vest":    "quy định bắt buộc mặc áo phản quang bảo hộ lao động",
        }
        seen_pages: set = set()
        blocks: list[str] = []
        for vtype in set(violation_types):
            query = _VIOLATION_QUERIES.get(vtype)
            if not query:
                continue
            for chunk in retrieve(query, k=2):
                key = (chunk["source"], chunk["page"])
                if key in seen_pages:
                    continue
                seen_pages.add(key)
                label = (
                    f"Dieu {chunk['dieu']}" if chunk["source"] == "dieu_khoan"
                    else f"Phu luc I, trang {chunk['page']}"
                )
                blocks.append(f"[Thong tu 25/2022 - {label}]\n{chunk['text']}")
        regulation_context = "\n\n".join(blocks)
    except Exception as exc:
        logger.debug("report_agent: RAG retrieval failed: %s", exc)

    # Build prompt
    quality_section = ""
    if quality.get("labeled_events", 0) > 0:
        quality_section = (
            f"\nLabel quality ({quality['labeled_events']} labeled events, "
            f"coverage={quality['label_coverage_note']}):\n"
            f"  Precision: {quality['precision']}\n"
            f"  Recall:    {quality['recall']}\n"
            f"  TP={quality['item_tp']} FP={quality['item_fp']} "
            f"FN={quality['item_fn']} TN={quality['item_tn']}\n"
        )

    systemic_section = ""
    if flags:
        systemic_section = "\nSystemic risk flags:\n" + "\n".join(f"  - {f}" for f in flags) + "\n"

    regulation_section = (
        f"\nRelevant regulations (Thong tu 25/2022/TT-BLDTBXH):\n{regulation_context}\n"
        if regulation_context else ""
    )

    peak_h = summary.get("peak_hour")
    prompt = (
        f"Date: {report_date}\n"
        f"Total violation events  : {summary.get('total_events', 0)}\n"
        f"Missing hardhat         : {summary.get('none_hardhat', 0)}\n"
        f"Missing vest            : {summary.get('none_vest', 0)}\n"
        f"Total violations        : {summary.get('total_violations', 0)}\n"
        f"Peak hour               : {f'{peak_h}:00 ICT' if peak_h is not None else 'N/A (no events)'}\n"
        f"Top offender            : {summary.get('top_offender_id', 'N/A')} "
        f"({summary.get('top_offender_count', 0)} violations in that session)\n"
        f"Avg detection confidence: {summary.get('avg_confidence', 0)}\n"
        f"Shift context           : {'night shift' if peak_h is not None and (peak_h >= 20 or peak_h <= 5) else 'day shift'}\n"
        f"{systemic_section}{quality_section}{regulation_section}"
    )

    # LLM call
    narrative = ""
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        narrative = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("report_agent: LLM call failed: %s", exc)
        narrative = f"[Report generation failed: {exc}]"

    # Persist to Supabase
    try:
        from src.infrastructure.supabase import get_supabase_service_client
        client = get_supabase_service_client()
        if client:
            client.table("daily_reports").upsert(
                {
                    "report_date":    report_date,
                    "summary":        summary,
                    "narrative":      narrative,
                    "quality":        quality,
                    "systemic_flags": flags,
                    "engine":         "langgraph",
                },
                on_conflict="report_date",
            ).execute()
            logger.info("report_agent: persisted to daily_reports for %s", report_date)
    except Exception as exc:
        logger.warning("report_agent: Supabase persist failed: %s", exc)

    return {**state, "narrative": narrative, "regulation_context": regulation_context}


# ── Graph assembly ─────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(ReportState)
    g.add_node("data_agent",        data_agent)
    g.add_node("pattern_agent",     pattern_agent)
    g.add_node("reliability_agent", reliability_agent)
    g.add_node("report_agent",      report_agent)

    g.set_entry_point("data_agent")
    g.add_edge("data_agent",        "pattern_agent")
    g.add_edge("pattern_agent",     "reliability_agent")
    g.add_edge("reliability_agent", "report_agent")
    g.add_edge("report_agent",      END)

    return g.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public entrypoint ─────────────────────────────────────────────────────────

async def run_report_pipeline(report_date: str | None = None) -> dict:
    """
    Run the full multi-agent report pipeline for the given date (ICT).
    report_date: 'YYYY-MM-DD'. Defaults to today ICT.
    Returns the final ReportState dict.
    """
    if report_date is None:
        report_date = datetime.now(ICT).strftime("%Y-%m-%d")

    initial_state: ReportState = {
        "report_date":       report_date,
        "violations":        [],
        "summary":           {},
        "recurring_trackers": [],
        "systemic_flags":    [],
        "quality":           {},
        "label_coverage_note": "no_data",
        "narrative":         "",
        "regulation_context": "",
    }

    graph = _get_graph()
    # LangGraph compiled graphs are synchronous; invoke directly
    import asyncio
    result = await asyncio.get_running_loop().run_in_executor(None, graph.invoke, initial_state)
    return result
