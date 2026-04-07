"""
Daily pipeline entrypoint for GitHub Actions.

Order:
  1. Second opinion batch — auto-label uncertain events for today.
  2. LangGraph report    — multi-agent daily report with quality metrics.
  3. S3 upload           — export confirmed labels to S3.
  4. Drift monitor       — scout disagreement rate and retrain trigger.
  5. Cleanup             — purge expired crops from Supabase Storage.
  6. Telegram notify     — send summary to ops channel.

Run:
    python run_daily_pipeline.py [YYYY-MM-DD]

Date defaults to today ICT. Pass an explicit date for backfills.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

ICT = timezone(timedelta(hours=7))


def _date_arg() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return datetime.now(ICT).strftime("%Y-%m-%d")


async def main() -> None:
    date_str = _date_arg()
    print(f"[daily_pipeline] Running for date: {date_str}")

    # ── Step 1: Second opinion batch ──────────────────────────────
    print("[daily_pipeline] Step 1: second opinion batch")
    try:
        from src.api.second_opinion import run_second_opinion_batch
        so_result = await run_second_opinion_batch(date_str)
        print(f"[daily_pipeline] Second opinion: {so_result}")
    except Exception as exc:
        print(f"[daily_pipeline] Second opinion failed (non-fatal): {exc}")
        so_result = {}

    # ── Step 2: LangGraph report ───────────────────────────────────
    print("[daily_pipeline] Step 2: LangGraph report")
    try:
        from src.api.report_pipeline import run_report_pipeline
        report = await run_report_pipeline(date_str)
        print(f"[daily_pipeline] Report generated for {date_str}")
        print(f"[daily_pipeline] Summary: {report.get('summary', {})}")
    except Exception as exc:
        print(f"[daily_pipeline] Report failed: {exc}")
        raise

    # ── Step 3: S3 upload ─────────────────────────────────────────
    print("[daily_pipeline] Step 3: S3 upload")
    s3_result: dict = {}
    try:
        s3_bucket = os.environ.get("S3_BUCKET", "")
        if s3_bucket:
            from src.flywheel.s3_uploader import upload_date_to_s3
            s3_result = upload_date_to_s3(date_str)
            print(f"[daily_pipeline] S3 upload: {s3_result}")
        else:
            print("[daily_pipeline] S3 upload skipped (S3_BUCKET not set)")
    except Exception as exc:
        print(f"[daily_pipeline] S3 upload failed (non-fatal): {exc}")

    # ── Step 4: Drift monitor ──────────────────────────────────────
    print("[daily_pipeline] Step 4: drift monitor")
    drift_result: dict = {}
    try:
        from src.flywheel.drift_monitor import run_drift_monitor
        drift_result = run_drift_monitor(date_str)
        print(f"[daily_pipeline] Drift: {drift_result}")
    except Exception as exc:
        print(f"[daily_pipeline] Drift monitor failed (non-fatal): {exc}")

    # ── Step 5: Cleanup expired crops ──────────────────────────────
    print("[daily_pipeline] Step 5: cleanup expired crops")
    try:
        from src.api.label_service import cleanup_expired_crops
        cleanup_result = cleanup_expired_crops()
        print(f"[daily_pipeline] Cleanup: {cleanup_result}")
    except Exception as exc:
        print(f"[daily_pipeline] Cleanup failed (non-fatal): {exc}")

    # ── Step 6: Telegram notification ─────────────────────────────
    print("[daily_pipeline] Step 6: Telegram notification")
    try:
        _send_telegram(date_str, report, so_result, drift_result)
    except Exception as exc:
        print(f"[daily_pipeline] Telegram failed (non-fatal): {exc}")


def _send_telegram(date_str: str, report: dict, so_result: dict, drift_result: dict | None = None) -> None:
    import requests

    token   = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    summary = report.get("summary", {})
    quality = report.get("quality", {})
    flags   = report.get("systemic_flags", [])
    narrative = report.get("narrative", "")

    prec = quality.get("precision")
    rec  = quality.get("recall")
    prec_str = f"{prec:.0%}" if prec is not None else "n/a"
    rec_str  = f"{rec:.0%}" if rec is not None else "n/a"
    quality_line = f"Quality ({quality.get('label_coverage_note', 'no data')}): Precision {prec_str} | Recall {rec_str}"
    quality_counts_line = (
        f"\nQuality counts: TP {quality.get('item_tp', 0)} | FP {quality.get('item_fp', 0)} | "
        f"FN {quality.get('item_fn', 0)} | TN {quality.get('item_tn', 0)}"
    )

    so_line = ""
    if so_result.get("processed"):
        so_line = (
            f"\nSecond opinion: {so_result['processed']} uncertain events processed, "
            f"{so_result['stored']} auto-labeled."
        )

    flags_text = ""
    if flags:
        flags_text = "\nFlags:\n" + "\n".join(f"  - {f}" for f in flags[:3])

    drift_line = ""
    drift_counts_line = ""
    if drift_result:
        rolling = drift_result.get("rolling_7d_rate")
        rate    = drift_result.get("disagreement_rate", 0.0)
        scout_rate = drift_result.get("scout_disagreement_rate")
        trigger = drift_result.get("retrain_triggered", False)
        rate_str    = f"{rate:.0%}"
        rolling_str = f"{rolling:.0%}" if rolling is not None else "n/a"
        drift_line  = f"\nDrift (human): {rate_str} today | 7d avg {rolling_str}"
        if scout_rate is not None:
            drift_line += f" | scout {scout_rate:.0%}"
        human_dis = int(drift_result.get("human_disagreements", 0) or 0)
        human_tot = int(drift_result.get("human_total_confirmed", 0) or 0)
        scout_dis = int(drift_result.get("scout_disagreements", 0) or 0)
        scout_tot = int(drift_result.get("scout_total_auto", 0) or 0)
        drift_counts_line = f"\nDrift counts: human {human_dis}/{human_tot} | scout {scout_dis}/{scout_tot}"
        if trigger:
            drift_line += " [RETRAIN TRIGGERED]"

    message = (
        f"PPE Vision - Daily Report ({date_str})\n"
        f"{'=' * 40}\n"
        f"Events       : {summary.get('total_events', 0)}\n"
        f"Hardhat miss : {summary.get('none_hardhat', 0)}\n"
        f"Vest miss    : {summary.get('none_vest', 0)}\n"
        f"Peak hour    : {summary.get('peak_hour', 'N/A')}\n"
        f"{quality_line}"
        f"{quality_counts_line}"
        f"{so_line}"
        f"{drift_line}"
        f"{drift_counts_line}"
        f"{flags_text}\n"
        f"{'=' * 40}\n\n"
        f"{narrative}"
    )

    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=10,
    )
    resp.raise_for_status()
    print("[daily_pipeline] Telegram notification sent.")


if __name__ == "__main__":
    asyncio.run(main())
