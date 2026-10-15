"""Explicit CI notification; no notification on import or missing webhook."""

import asyncio
import os
from pathlib import Path

from ci_pipeline import read_json
from slack_alerts import SlackAlertError, build_slack_message, send_slack_alert


async def main():
    if not os.environ.get("SLACK_WEBHOOK_URL", "").strip():
        print("Optional Slack secret not configured; no message sent.")
        return
    state = read_json("ci_output/status.json")
    if state["status"] == "NOT_APPLICABLE":
        return
    url = os.environ.get("REPORT_ARTIFACT_URL", "")
    if state["status"] in ("PASS", "WARN", "FAIL") and Path("ci_output/scored.json").exists():
        payload = build_slack_message(read_json("ci_output/scored.json"), read_json("datasets/golden_v1.json"),
                                      previous=read_json("ci/baseline.json"),
                                      previous_dataset=read_json("ci/baseline_dataset.json"), report_url=url)
    else:
        payload = {"text": "INCOMPLETE | PR evaluation did not complete. This is not a PASS.\nReport: " + url}
    try:
        await send_slack_alert(payload)
        print("Slack delivery confirmed.")
    except SlackAlertError:
        # Notification failure must not suppress the critical-regression gate.
        print("::warning::Slack delivery not confirmed; no automatic retry.")


if __name__ == "__main__":
    asyncio.run(main())
