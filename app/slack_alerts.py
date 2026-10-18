"""Slack Incoming Webhook alerts from saved evaluations; no LLM calls."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
import httpx

from app.comparison import compare_runs
from app.html_report import _summary
from app.thresholds import ThresholdConfig
from app.drift_detection import DriftConfig, detect_slow_drift


class SlackAlertError(RuntimeError):
    """Safe delivery error without webhook URLs or response bodies."""


def _report_url(value: str) -> str:
    try:
        url = urlsplit(value)
        host = url.hostname
        if url.scheme != "https" or not host or url.username or url.password or any(c in value for c in "\r\n\t "):
            raise ValueError
        if host.lower() in ("localhost", "localhost.localdomain") or host.lower().endswith((".localhost", ".local")):
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("Report URL must be an accessible HTTPS URL, without embedded credentials.") from None
    return value


def build_slack_message(current: dict, dataset: dict, *, previous: dict | None = None,
                        previous_dataset: dict | None = None, report_url: str | None = None,
                        thresholds: ThresholdConfig | None = None,
                        history: list | None = None, drift_config: DriftConfig | None = None) -> dict:
    """Build Block Kit payload offline; no baseline must never imply PASS."""
    diff = compare_runs(current, previous, current_dataset=dataset,
                        previous_dataset=previous_dataset, thresholds=thresholds)
    stats = _summary(current, dataset)
    severity = diff["threshold_assessment"]["severity"]
    status = {"ok": "PASS", "warning": "WARN", "critical": "FAIL",
              "not_evaluated": "NOT EVALUATED"}[severity]
    gaps = len(diff["unevaluable_cases"]) + len(diff["judge_incompatible_cases"])
    if previous is not None and (gaps or stats["evaluable"] < stats["count"]) and status == "PASS":
        status = "WARN"
    def rate(value):
        return "N/A" if value is None else f"{value * 100:.1f}%"
    lines = [f"Kapsam: {stats['count']}/{len(dataset['cases'])} kayıt; "
             f"değerlendirilebilir: {stats['evaluable']}; pass: {stats['passed']}.",
             f"Mevcut pass oranı: {rate(stats['pass_rate'])}; kategori doğruluğu: {rate(stats['accuracy'])}."]
    if history is not None:
        drift = detect_slow_drift(current, dataset, history=history, config=drift_config, thresholds=thresholds)
        if drift["slow_drift"]:
            if status != "FAIL":
                status = "WARN"
            lines.append(f"SLOW DRIFT: Son {drift['config']['window_size']} çalışmanın pass ortalaması "
                         f"{rate(drift['moving_average'])}; eşik {rate(drift['config']['minimum_average_pass_rate'])}.")
        else:
            lines.append(f"Slow drift: {drift['status']}; ardışık uygun çalışma: {drift['available_consecutive_runs']}.")
    if previous is None:
        lines.append("Baseline yok; gerileme, iyileşme ve oran farkı hesaplanamadı.")
    else:
        metric = diff["overall_pass_rate"]
        delta = metric["delta_percentage_points"]
        delta_text = "N/A" if delta is None else f"{delta:+.2f} yüzde puan"
        lines.extend([
            f"{len(diff['regressions'])} gerileme; {len(diff['improvements'])} iyileşme.",
            f"Pass oranı: {rate(metric['previous_rate'])} → {rate(metric['current_rate'])} ({delta_text}); ortak payda: {metric['compared_cases']}.",
            f"Değerlendirilemeyen/uyumsuz judge: {gaps}; değişen referans: {len(diff['changed_cases'])}; "
            f"eklenen: {len(diff['added_cases'])}; kaldırılan: {len(diff['removed_cases'])}.",
        ])
        for category, metric in diff["per_category_accuracy"].items():
            if metric["compared_cases"]:
                lines.append(f"{category} doğruluğu: {rate(metric['previous_rate'])} → {rate(metric['current_rate'])} "
                             f"({metric['delta_percentage_points']:+.2f} yüzde puan; n={metric['compared_cases']}).")
    if stats["judge"] is not None:
        lines.append(f"Ortalama judge: {stats['judge']:.2f}/5.")
    if current.get("live_test", {}).get("report_kind") == "consolidated_live_subset":
        lines.append("Başarılı sonuçlar birden fazla oturumdan birleştirilmiştir; önceki hatalar HTML raporundadır.")
    title = f"{status} | Model Regression Detection"
    body = "\n".join(lines)
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": title}},
        {"type": "section", "text": {"type": "plain_text", "text": body}},
        {"type": "context", "elements": [{"type": "plain_text", "text": (
            f"Prompt: {current.get('prompt_version', 'N/A')} | Model: {current.get('model', 'N/A')} | "
            f"Başlangıç: {current.get('started_at', 'N/A')} | Veri seti: {current.get('dataset_version', 'N/A')}"
        )[:2000]}]},
    ]
    if report_url:
        report_url = _report_url(report_url)
        blocks.append({"type": "actions", "elements": [{"type": "button", "action_id": "open_diff_report",
            "text": {"type": "plain_text", "text": "Tam HTML raporunu aç"}, "url": report_url}]})
    else:
        blocks.append({"type": "context", "elements": [{"type": "plain_text",
                      "text": "Rapor bağlantısı henüz yapılandırılmadı; HTML yalnızca yerel dosyada mevcut."}]})
    return {"text": title + "\n" + body + ("\nRapor: " + report_url if report_url else "\nRapor bağlantısı yapılandırılmadı."),
            "blocks": blocks, "unfurl_links": False, "unfurl_media": False}


async def send_slack_alert(payload: dict, *, env_file: str | Path | None = None,
                           transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """Send once, without redirects or retries; ambiguous failures are not retried."""
    values = dotenv_values(env_file if env_file is not None else Path(__file__).resolve().parents[1] / ".env")
    webhook = os.environ.get("SLACK_WEBHOOK_URL", values.get("SLACK_WEBHOOK_URL") or "").strip()
    try:
        url = urlsplit(webhook)
        if (url.scheme != "https" or url.hostname != "hooks.slack.com" or url.port not in (None, 443)
                or url.username or url.password or url.query or url.fragment
                or not url.path.startswith("/services/") or len(url.path.split("/")) != 5
                or any(not part for part in url.path.split("/")[2:]) or any(c.isspace() for c in webhook)):
            raise ValueError
    except (ValueError, TypeError):
        raise SlackAlertError("SLACK_WEBHOOK_URL is missing or invalid.") from None
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False, transport=transport) as client:
            response = await client.post(webhook, json=payload)
    except httpx.TimeoutException:
        raise SlackAlertError("Slack delivery timed out; delivery is unknown. No retry attempted.") from None
    except httpx.TransportError:
        raise SlackAlertError("Slack connection failed; delivery is unknown. No retry attempted.") from None
    if response.status_code != 200 or response.text.strip() != "ok":
        raise SlackAlertError(f"Slack did not confirm delivery (HTTP {response.status_code}). No retry attempted.")
    return {"delivered": True, "http_status": 200}


def main():
    import asyncio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--previous-dataset", type=Path)
    parser.add_argument("--report-url")
    parser.add_argument("--send", action="store_true", help="Send one real Slack message; default is preview only.")
    args = parser.parse_args()
    if bool(args.previous) != bool(args.previous_dataset):
        parser.error("--previous and --previous-dataset are required together")
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig")) if path else None
    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    report_url = args.report_url or os.environ.get("HTML_REPORT_URL", values.get("HTML_REPORT_URL"))
    payload = build_slack_message(read(args.report), read(args.dataset), previous=read(args.previous),
                                 previous_dataset=read(args.previous_dataset), report_url=report_url)
    if args.send:
        try:
            print(json.dumps(asyncio.run(send_slack_alert(payload))))
        except SlackAlertError as exc:
            raise SystemExit(str(exc)) from None
    else:
        print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
