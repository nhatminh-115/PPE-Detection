"""
PPE Vision - Daily Report Service
Pulls violation logs from Supabase, generates narrative report via Groq,
then stores JSON summary to Supabase and sends Telegram notification.
"""

import os
import json
import requests
from datetime import date, datetime
from collections import defaultdict
from supabase import create_client, Client
from groq import Groq
from datetime import datetime, timezone, timedelta

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
    """Pull all violation rows for a given date."""
    response = (
        client.table("ppe_violations")
        .select("*")
        .gte("created_at", f"{report_date}T00:00:00+00:00")
        .lt("created_at", f"{report_date}T23:59:59+00:00")
        .execute()
    )
    return response.data or []


def aggregate(rows: list[dict]) -> dict:
    """
    Unpack JSONB violation arrays and compute summary statistics.

    Each row.violations is a list of dicts:
        {"confidence": float, "image_path": str,
         "tracker_id": int, "violation_type": str}
    """
    total_events = len(rows)
    violation_counts = defaultdict(int)   # violation_type -> count
    tracker_counts   = defaultdict(int)   # tracker_id     -> count
    hourly_counts    = defaultdict(int)   # hour (int)     -> count
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

    peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
    top_offender = max(tracker_counts, key=tracker_counts.get) if tracker_counts else None
    avg_confidence = round(confidence_sum / confidence_n, 4) if confidence_n else 0.0

    return {
        "total_events":      total_events,
        "none_hardhat":      violation_counts.get("none_hardhat", 0),
        "none_vest":         violation_counts.get("none_vest", 0),
        "total_violations":  sum(violation_counts.values()),
        "top_offender_id":   top_offender,
        "top_offender_count": tracker_counts.get(top_offender, 0) if top_offender else 0,
        "peak_hour":         peak_hour,
        "avg_confidence":    avg_confidence,
        "hourly_breakdown":  dict(hourly_counts),
        "tracker_breakdown": dict(tracker_counts),
    }


# ---------------------------------------------------------------------------
# LLM report generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a workplace safety analyst. Given a daily PPE violation summary,
write a concise safety report in 3 short paragraphs:
1. Overview of the day's violations.
2. Key risk areas (which violation type dominated, peak hours, repeat offenders).
3. Recommended corrective actions.

Be direct and professional. Do not use bullet points. Keep under 200 words.
"""

def generate_narrative(groq_client: Groq, summary: dict, report_date: date) -> str:
    prompt = f"""
Date: {report_date}
Total violation events: {summary['total_events']}
Missing hardhat violations: {summary['none_hardhat']}
Missing vest violations: {summary['none_vest']}
Total individual violations: {summary['total_violations']}
Peak hour: {summary['peak_hour']}:00
Most frequent offender (tracker ID): {summary['top_offender_id']} ({summary['top_offender_count']} violations)
Average detection confidence: {summary['avg_confidence']}
Hourly breakdown: {json.dumps(summary['hourly_breakdown'])}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_to_supabase(client: Client, report_date: date, summary: dict, narrative: str) -> None:
    client.table("daily_reports").upsert({
    "report_date": str(report_date),
    "summary":     summary,
    "narrative":   narrative,
    }, on_conflict="report_date").execute()


def send_telegram(summary: dict, narrative: str, report_date: date) -> None:
    token   = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    message = (
        f"PPE Vision - Daily Report ({report_date})\n"
        f"{'=' * 40}\n"
        f"Total events     : {summary['total_events']}\n"
        f"Missing hardhat  : {summary['none_hardhat']}\n"
        f"Missing vest     : {summary['none_vest']}\n"
        f"Peak hour        : {summary['peak_hour']}:00\n"
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

    summary   = aggregate(rows)
    narrative = generate_narrative(groq, summary, report_date)

    save_to_supabase(supabase, report_date, summary, narrative)
    send_telegram(summary, narrative, report_date)

    print(f"[report_service] Done. {summary['total_violations']} violations reported.")


if __name__ == "__main__":
    main()