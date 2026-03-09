"""
PPE Vision - Daily Report Service
Pulls violation logs from Supabase, generates narrative report via Groq,
with RAG-augmented QCVN regulation context.
"""

import os
import json
import requests
from datetime import date, datetime, timezone, timedelta
from collections import defaultdict
from supabase import create_client, Client
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

ICT = timezone(timedelta(hours=7))

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def get_groq() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def fetch_today_violations(client: Client, report_date: date) -> list[dict]:
    response = (
        client.table("ppe_violations")
        .select("*")
        .gte("created_at", f"{report_date}T00:00:00+00:00")
        .lt("created_at", f"{report_date}T23:59:59+00:00")
        .execute()
    )
    return response.data or []


def aggregate(rows: list[dict]) -> dict:
    total_events     = len(rows)
    violation_counts = defaultdict(int)
    tracker_counts   = defaultdict(int)
    hourly_counts    = defaultdict(int)
    confidence_sum   = 0.0
    confidence_n     = 0

    for row in rows:
        hour = datetime.fromisoformat(row["created_at"]).astimezone(ICT).hour
        hourly_counts[hour] += 1

        for v in row.get("violations", []):
            violation_counts[v["violation_type"]] += 1
            tracker_counts[str(v["tracker_id"])] += 1
            confidence_sum += v.get("confidence", 0.0)
            confidence_n   += 1

    peak_hour    = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
    top_offender = max(tracker_counts, key=tracker_counts.get) if tracker_counts else None
    avg_conf     = round(confidence_sum / confidence_n, 4) if confidence_n else 0.0

    return {
        "total_events":       total_events,
        "none_hardhat":       violation_counts.get("none_hardhat", 0),
        "none_vest":          violation_counts.get("none_vest", 0),
        "total_violations":   sum(violation_counts.values()),
        "top_offender_id":    top_offender,
        "top_offender_count": tracker_counts.get(top_offender, 0) if top_offender else 0,
        "peak_hour":          peak_hour,
        "avg_confidence":     avg_conf,
        "hourly_breakdown":   dict(hourly_counts),
        "tracker_breakdown":  dict(tracker_counts),
    }


# ---------------------------------------------------------------------------
# RAG context retrieval
# ---------------------------------------------------------------------------

# Map violation types to Vietnamese search queries for better retrieval
_VIOLATION_QUERIES = {
    "none_hardhat": "quy định bắt buộc đội mũ an toàn công nghiệp công nhân xây dựng",
    "none_vest":    "quy định bắt buộc mặc áo phản quang bảo hộ lao động",
}

def fetch_regulation_context(violation_types: list[str]) -> str:
    """
    Query ChromaDB for relevant QCVN regulations based on violation types.
    Returns formatted context string to inject into LLM prompt.
    Falls back gracefully if RAG index is unavailable.
    """
    try:
        from rag.query import retrieve
    except ImportError:
        print("[report_service] RAG module not found, skipping regulation context.")
        return ""

    seen_pages = set()
    context_blocks = []

    for vtype in set(violation_types):
        query = _VIOLATION_QUERIES.get(vtype)
        if not query:
            continue

        chunks = retrieve(query, k=2)
        for chunk in chunks:
            page_key = (chunk["source"], chunk["page"])
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)

            label = (
                f"Dieu {chunk['dieu']}" if chunk["source"] == "dieu_khoan"
                else f"Phu luc I, trang {chunk['page']}"
            )
            context_blocks.append(
                f"[Thong tu 25/2022 - {label}]\n{chunk['text']}"
            )

    if not context_blocks:
        return ""

    return "\n\n".join(context_blocks)


# ---------------------------------------------------------------------------
# LLM report generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a workplace safety analyst in Vietnam.
Given a daily PPE violation summary and relevant Vietnamese labor regulations,
write a concise safety report in 3 short paragraphs:
1. Overview of the day's violations.
2. Key risk areas (violation type, peak hours, repeat offenders).
3. Recommended corrective actions, citing specific regulations where available.

Be direct and professional. Do not use bullet points. Keep under 250 words."""


def generate_narrative(
    groq_client: Groq,
    summary: dict,
    report_date: date,
    regulation_context: str = "",
) -> str:
    regulation_section = (
        f"\nRelevant regulations (Thong tu 25/2022/TT-BLDTBXH):\n{regulation_context}\n"
        if regulation_context else ""
    )

    prompt = f"""Date: {report_date}
Total violation events  : {summary['total_events']}
Missing hardhat         : {summary['none_hardhat']}
Missing vest            : {summary['none_vest']}
Total violations        : {summary['total_violations']}
Peak hour               : {summary['peak_hour']}:00 ICT
Top offender tracker ID : {summary['top_offender_id']} ({summary['top_offender_count']} violations)
Avg detection confidence: {summary['avg_confidence']}
Hourly breakdown        : {json.dumps(summary['hourly_breakdown'])}
{regulation_section}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_to_supabase(
    client: Client,
    report_date: date,
    summary: dict,
    narrative: str,
) -> None:
    client.table("daily_reports").upsert(
        {
            "report_date": str(report_date),
            "summary":     summary,
            "narrative":   narrative,
        },
        on_conflict="report_date",
    ).execute()


def send_telegram(summary: dict, narrative: str, report_date: date) -> None:
    token   = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    message = (
        f"PPE Vision - Daily Report ({report_date})\n"
        f"{'=' * 40}\n"
        f"Total events     : {summary['total_events']}\n"
        f"Missing hardhat  : {summary['none_hardhat']}\n"
        f"Missing vest     : {summary['none_vest']}\n"
        f"Peak hour        : {summary['peak_hour']}:00 ICT\n"
        f"Top offender ID  : {summary['top_offender_id']} "
        f"({summary['top_offender_count']} violations)\n"
        f"{'=' * 40}\n\n"
        f"{narrative}"
    )

    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=10,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    report_date = date.today()
    print(f"[report_service] Generating report for {report_date}")

    supabase = get_supabase()
    groq     = get_groq()

    rows = fetch_today_violations(supabase, report_date)
    if not rows:
        print("[report_service] No violations found today. Skipping.")
        return

    summary = aggregate(rows)

    # Collect unique violation types from today's data
    violation_types = list({
        v["violation_type"]
        for row in rows
        for v in row.get("violations", [])
    })

    print(f"[report_service] Fetching regulation context for: {violation_types}")
    regulation_context = fetch_regulation_context(violation_types)
    if regulation_context:
        print("[report_service] RAG context retrieved successfully.")
    else:
        print("[report_service] No RAG context, generating without regulations.")

    narrative = generate_narrative(groq, summary, report_date, regulation_context)

    save_to_supabase(supabase, report_date, summary, narrative)
    send_telegram(summary, narrative, report_date)

    print(f"[report_service] Done. {summary['total_violations']} violations reported.")


if __name__ == "__main__":
    main()