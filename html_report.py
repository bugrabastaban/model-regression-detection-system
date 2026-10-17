"""Standalone HTML reports from saved scored results; no network calls."""

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path
from statistics import mean

from comparison import CATEGORIES, _state, _token_counts, compare_runs
from drift_detection import DriftConfig, detect_slow_drift


def _e(value):
    return escape(str(value)) if value is not None else "—"


def _percent(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def _token_value(value, *, delta=False):
    return "bilinmiyor" if value is None else f"{value:+d}" if delta else str(value)


def _token_cell(counts):
    return '<br>'.join(f'{label}: {_token_value(counts[key])}' for key, label in (
        ("input_tokens", "Input"), ("output_tokens", "Output"), ("total_tokens", "Total")))


def _summary(report, dataset):
    refs = {case["id"]: case for case in dataset["cases"]}
    states = [_state(row, refs[row["case_id"]]) for row in report["results"]]
    valid = [state for state in states if state["pass"] is not None]
    categories = [s["category_match"] for s in states if s["category_match"] is not None]
    ratings = [s["summary_relevance"] for s in states if s["summary_relevance"] is not None]
    return {"count": len(states), "evaluable": len(valid),
            "passed": sum(s["pass"] for s in valid),
            "pass_rate": mean(s["pass"] for s in valid) if valid else None,
            "accuracy": mean(categories) if categories else None,
            "judge": mean(ratings) if ratings else None}


def _run_key(report):
    """Use run_id, or immutable run metadata for legacy reports without IDs.

    Legacy producers must use distinct metadata (especially started_at) or add
    run_id to distinguish separate runs. Outputs are not part of run identity.
    """
    run_id = report.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return ("run_id", run_id)
    metadata = {key: report.get(key) for key in (
        "started_at", "prompt_version", "prompt_timestamp", "model",
        "dataset_version", "judge_rubric_version", "commit_sha")}
    return ("metadata", json.dumps(metadata, sort_keys=True, ensure_ascii=True))


def _unique_runs(entries):
    """Deduplicate before windowing; the last supplied copy takes precedence."""
    return list({_run_key(report): (report, snapshot) for report, snapshot in entries}.values())


def _trend(current, dataset, history, last_n):
    """Only graph identical evaluated cohorts and judge configurations."""
    candidates = []
    ids = {row["case_id"] for row in current["results"]}
    for report, snapshot in _unique_runs(history):
        if _run_key(report) == _run_key(current):
            continue
        diff = compare_runs(current, report, current_dataset=dataset, previous_dataset=snapshot)
        same = ids == {row["case_id"] for row in report["results"]}
        if diff["status"] == "compared" and same and not any(diff[key] for key in
                            ("changed_cases", "unevaluable_cases", "judge_incompatible_cases")):
            candidates.append((report, snapshot))
    candidates.append((current, dataset))
    candidates.sort(key=lambda pair: datetime.fromisoformat(pair[0]["started_at"]))
    candidates = candidates[-last_n:]
    points = []
    labels = []
    for index, (report, snapshot) in enumerate(candidates):
        stats = _summary(report, snapshot)
        if stats["pass_rate"] is None:
            continue
        x = 80 + index * (760 / max(1, len(candidates) - 1))
        y = 170 - stats["pass_rate"] * 140
        points.append((x, y))
        labels.append(f'<text x="{x}" y="200" text-anchor="middle">{index + 1}</text>')
    if not points:
        return '<p class="muted">Çizilecek puan verisi yok.</p>'
    dots = ''.join(f'<circle cx="{x}" cy="{y}" r="6"><title>{i+1}. kayıt: {_percent(_summary(*candidates[i])["pass_rate"])}</title></circle>' for i, (x, y) in enumerate(points))
    line = ' '.join(f'{x},{y}' for x, y in points)
    note = "Tek sonuç kaydı var; zaman içindeki değişim henüz ölçülemiyor." if len(candidates) < 2 else f"Son {len(candidates)} karşılaştırılabilir kayıt; aynı örnekler ve judge yapılandırması."
    entries = ''.join(f'<li>{_e(report["started_at"])} · prompt {_e(report.get("prompt_version"))} · {_percent(_summary(report, snapshot)["pass_rate"])}</li>' for report, snapshot in candidates)
    return f'''<p class="muted">{note} Geçmiş için üst sınır: {last_n}.</p>
    <svg viewBox="0 0 920 220" role="img" aria-label="Pass oranı trendi">
    <text x="15" y="35">100%</text><text x="35" y="175">0%</text>
    <path d="M70 30H870 M70 170H870" class="grid"/>
    <polyline points="{line}"/>{dots}{''.join(labels)}</svg><ol>{entries}</ol>'''


def render_report(current: dict, dataset: dict, *, previous: dict | None = None,
                  previous_dataset: dict | None = None, history: list | None = None,
                  last_n: int = 10, drift_config: DriftConfig | None = None) -> str:
    """Render scalable, escaped HTML; previous run and history are explicit."""
    if type(last_n) is not int or last_n < 1:
        raise ValueError("last_n must be a positive integer.")
    diff = compare_runs(current, previous, current_dataset=dataset, previous_dataset=previous_dataset)
    stats = _summary(current, dataset)
    # Coverage is relative to the supplied snapshot, not an inferred full dataset.
    # Evaluable includes both PASS and FAIL; comparison uses the existing paired
    # pass denominator, which can be smaller due to baseline/reference/judge gaps.
    expected_count = len(dataset["cases"])

    def coverage(count):
        label = "Veri yok" if expected_count == 0 else "Tam kapsam" if count == expected_count else "Kısmi kapsam"
        return f"{label} ({count} / {expected_count} kayıt)"

    if previous is None:
        comparison_coverage = "Baseline yok; hesaplanmadı."
    elif diff["status"] == "incompatible_pass_policy":
        comparison_coverage = "Pass politikaları uyumsuz; hesaplanmadı."
    else:
        comparison_coverage = coverage(diff["overall_pass_rate"]["compared_cases"])
    coverage_html = (f'<p>{stats["count"]} / {expected_count} kayıt · Sağlanan veri seti snapshot’ına göre:</p>'
                     f'<p>İşlenen kapsam: {coverage(stats["count"])}<br>'
                     f'Değerlendirilebilir kapsam: {coverage(stats["evaluable"])}<br>'
                     f'Baseline ile pass karşılaştırması: {comparison_coverage}</p>')
    if history is not None:
        # Saved copies of current/history are not additional drift observations.
        history = [(report, snapshot) for report, snapshot in _unique_runs(history)
                   if _run_key(report) != _run_key(current)]
        drift = detect_slow_drift(current, dataset, history=history, config=drift_config)
        drift_html = (f"<p><b>{_e(drift['status'])}</b> · Son {drift['config']['window_size']} çalışma "
                      f"ortalaması: {_percent(drift['moving_average'])} · Eşik: "
                      f"{_percent(drift['config']['minimum_average_pass_rate'])} · "
                      f"Ardışık uygun çalışma: {drift['available_consecutive_runs']}</p>")
    else:
        drift_html = '<p class="muted">Çalışma geçmişi sağlanmadı; slow drift değerlendirilmedi.</p>'
    refs = {case["id"]: case for case in dataset["cases"]}
    old_rows = {row["case_id"]: row for row in previous["results"]} if previous else {}
    rows = {row["case_id"]: row for row in current["results"]}
    live = current.get("live_test", {})
    consolidated = live.get("report_kind") == "consolidated_live_subset"
    provenance = "İki canlı oturumdaki başarılı sonuçlar birleştirilmiştir; tek kesintisiz çalışma değildir." if consolidated else "Kaydedilmiş değerlendirme sonuçları."
    metadata = ''.join(f'<div><dt>{label}</dt><dd>{_e(current.get(key))}</dd></div>' for key, label in
                       (("prompt_version", "Prompt"), ("model", "Sınıflandırıcı"),
                        ("started_at", "Başlangıç (ISO 8601)"), ("dataset_version", "Veri seti"),
                        ("judge_rubric_version", "Judge rubriği")))
    judge_models = ', '.join(sorted({str(r["scores"]["judge"].get("model")) for r in rows.values() if r["scores"]["judge"].get("model")}))
    change = diff["overall_pass_rate"]
    delta = "Karşılaştırma yok" if change["delta_percentage_points"] is None else f'{change["delta_percentage_points"]:+.2f} yüzde puan'
    cards = ''.join(f'<article><span>{title}</span><strong>{value}</strong><small>{subtitle}</small></article>' for title, value, subtitle in (
        ("Pass oranı", _percent(stats["pass_rate"]), f'{stats["passed"]}/{stats["evaluable"]} değerlendirilebilir kayıt'),
        ("Kategori doğruluğu", _percent(stats["accuracy"]), "Geçerli sınıflandırmalar"),
        ("Ortalama judge", f'{stats["judge"]:.2f}/5' if stats["judge"] is not None else "—", _e(judge_models)),
        ("Baseline farkı", delta, _e(diff["threshold_assessment"]["severity"]))))
    if diff["status"] == "incompatible_pass_policy":
        comparison = '<p class="notice">incompatible_pass_policy: Pass politikaları uyumsuz; karşılaştırma ve eşikler değerlendirilmedi.</p>'
    elif previous is None:
        comparison = '<p class="notice">Baseline yok: bu sonuçlar başlangıç referansı olarak sunuluyor. Gerileme veya iyileşme henüz hesaplanamaz.</p>'
    else:
        comparison = f'''<p>Baseline: {_e(previous.get("started_at"))} · {_e(previous.get("model"))} · prompt {_e(previous.get("prompt_version"))}</p>
        <p>Ortak pass paydası: {change["compared_cases"]} · Önceki: {_percent(change["previous_rate"])} → Yeni: {_percent(change["current_rate"])} · {delta}</p>
        <p>Gerileme: {len(diff["regressions"])} · İyileşme: {len(diff["improvements"])} · Değişen referans: {len(diff["changed_cases"])} · Değerlendirilemeyen: {len(diff["unevaluable_cases"])} · Uyumsuz judge: {len(diff["judge_incompatible_cases"])}</p>'''
    def output_cell(row):
        output = row.get("output") or {}
        return f'<b>{_e(output.get("category"))}</b><p>{_e(output.get("summary"))}</p><small>Judge: {_e(row["scores"].get("summary_relevance"))}/5</small>'
    regressions = ''.join(f'<tr><td>{_e(key)}<p>{_e(refs[key]["input"])}</p><small>Beklenen: {_e(refs[key]["expected_category"])}</small></td><td>{output_cell(old_rows[key])}</td><td>{output_cell(rows[key])}</td></tr>' for key in diff["regressions"])
    regression_table = f'<div class="scroll"><table><thead><tr><th>Kayıt / e-posta</th><th>Eski çıktı</th><th>Yeni çıktı</th></tr></thead><tbody>{regressions}</tbody></table></div>' if regressions else '<p class="muted">Karşılaştırma verisi yok.</p>' if previous is None else '<p class="muted">Karşılaştırılabilir kayıtlarda pass → fail değişimi yok.</p>'
    case_rows = []
    usage = {key: {provider: _token_counts(row, provider) for provider in ("gemini", "groq")}
             for key, row in rows.items()}
    token_coverage = '; '.join(
        f'{label}: ' + ', '.join(
            f'{counter} {sum(usage[key][provider][counter + "_tokens"] is not None for key in rows)}/{len(rows)}'
            for counter in ("input", "output", "total"))
        for provider, label in (("gemini", "Gemini"), ("groq", "Groq judge")))
    token_diff_rows = ''.join(
        f'<tr><td>{_e(entry["case_id"])}</td><td>{label}</td><td>{counter}</td>'
        f'<td>{_token_value(entry[provider]["previous"][field])}</td>'
        f'<td>{_token_value(entry[provider]["current"][field])}</td>'
        f'<td>{_token_value(entry[provider]["delta_tokens"][field], delta=True)}</td></tr>'
        for entry in diff["token_usage_comparisons"]
        for provider, label in (("gemini", "Gemini"), ("groq", "Groq judge"))
        for field, counter in (("input_tokens", "Input"), ("output_tokens", "Output"), ("total_tokens", "Total")))
    token_comparison = (
        '<div class="scroll"><table><thead><tr><th>Kayıt</th><th>Sağlayıcı</th><th>Sayaç</th>'
        '<th>Önceki</th><th>Yeni</th><th>Fark (token)</th></tr></thead>'
        f'<tbody>{token_diff_rows}</tbody></table></div>' if token_diff_rows else
        '<p class="muted">Token karşılaştırması için uygun baseline kaydı yok.</p>')
    for key, row in rows.items():
        s = row["scores"]; j = s["judge"]; state = _state(row, refs[key])
        verdict = "PASS" if state["pass"] is True else "FAIL" if state["pass"] is False else "DEĞERLENDİRİLEMEDİ"
        latency = row.get("elapsed_seconds")
        case_rows.append(f'''<tr><td><b>{_e(key)}</b><details><summary>E-posta ve referans</summary><p>{_e(refs[key]["input"])}</p><p>Referans: {_e(refs[key]["ideal_summary"])}</p></details></td>
        <td>{_e(refs[key]["expected_category"])}</td><td>{output_cell(row)}</td><td>{verdict}<p>{_e(j.get("reason"))}</p></td>
        <td>{f"{latency:.2f} sn" if isinstance(latency, (int, float)) else "—"}</td>
        <td>{_token_cell(usage[key]["gemini"])}</td><td>{_token_cell(usage[key]["groq"])}</td></tr>''')
    history = list(history or [])
    if previous is not None:
        history.append((previous, previous_dataset))
    trend = _trend(current, dataset, history, last_n)
    category_rows = ''.join(f'<tr><td>{category}</td><td>{sum(refs[key]["expected_category"] == category for key in rows)}</td><td>{_percent(metric["previous_rate"])}</td><td>{_percent(metric["current_rate"])}</td><td>{_e(metric["delta_percentage_points"])}</td><td>{metric["compared_cases"]}</td></tr>' for category, metric in diff["per_category_accuracy"].items())
    attempts = f'<p>Kaynak oturumların toplam gönderimleri: Gemini {_e(live.get("gemini_http_attempts"))}, Groq {_e(live.get("groq_http_attempts"))}. Önceki başarısız deneme: {len(live.get("earlier_failed_attempts", []))} (HTTP 503; başarılı sonuç tablosuna dahil değildir).</p>' if consolidated else ''
    return f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
    <title>Model Regression · Değerlendirme Raporu</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f2f5f8;color:#182c40;font:15px/1.6 Segoe UI,Arial,sans-serif}}main{{max-width:1280px;margin:auto;padding:40px 28px}}header{{border-top:5px solid #087f8c;padding:24px 0}}h1{{font-size:34px;margin:8px 0}}h2{{font-size:22px}}.eyebrow{{color:#087f8c;font-weight:700;letter-spacing:2px}}.muted,small{{color:#52677c}}section{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:24px;margin-top:22px}}dl{{display:flex;flex-wrap:wrap;gap:22px}}dt{{font-size:12px;color:#52677c}}dd{{margin:0;overflow-wrap:anywhere}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:24px}}article{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:22px}}article strong{{display:block;font-size:29px;color:#087f8c}}article small{{display:block}}.notice{{background:#eaf5f6;border-left:4px solid #087f8c;padding:14px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:14px;text-align:left;vertical-align:top;border-bottom:1px solid #dce4ed}}th{{background:#edf3f7}}td p{{margin:8px 0}}.scroll{{overflow-x:auto}}details{{min-width:140px}}summary{{cursor:pointer;color:#087f8c}}svg{{width:100%;max-height:240px}}svg text{{font:13px Segoe UI;fill:#52677c}}polyline{{fill:none;stroke:#087f8c;stroke-width:3}}circle{{fill:#087f8c}}.grid{{stroke:#dce4ed;fill:none}}footer{{padding:24px 0;color:#52677c}}@media(max-width:800px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:20px 14px}}h1{{font-size:27px}}}}@media print{{body{{background:white}}section{{break-inside:avoid}}}}
    </style></head><body><main><header><div class="eyebrow">MODEL REGRESSION DETECTION SYSTEM</div><h1>Değerlendirme &amp; fark raporu</h1>
    {coverage_html}<p class="muted">{provenance}</p><dl>{metadata}</dl></header>
    <div class="cards">{cards}</div><section><h2>Baseline karşılaştırması</h2>{comparison}<p class="muted">Pass: doğru kategori ve judge ≥ 4/5. Eşikler düşüşte &gt;3 yüzde puan warning, &gt;8 yüzde puan critical. İstatistiksel anlamlılık testi değildir.</p></section>
    <section><h2>Gerileyen kayıtlar · eski / yeni</h2>{regression_table}</section>
    <section><h2>Kategori kapsamı ve karşılaştırma</h2><div class="scroll"><table><thead><tr><th>Kategori</th><th>Bu rapordaki kayıt</th><th>Eski doğruluk</th><th>Yeni doğruluk</th><th>Fark (yp)</th><th>Ortak payda</th></tr></thead><tbody>{category_rows}</tbody></table></div><p class="muted">Eski/yeni doğruluk yalnızca eşleştirilmiş kayıtları gösterir; baseline yoksa boştur.</p></section>
    <section><h2>İşlenen kayıtlar</h2><div class="scroll"><table><thead><tr><th>ID / girdi</th><th>Beklenen</th><th>Gemini çıktısı</th><th>Sonuç / judge gerekçesi</th><th>Süre</th><th>Gemini tokenları</th><th>Groq judge tokenları</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table></div><p class="muted">Bilinen sayaç kapsamı: {token_coverage}. “bilinmiyor”, eksik veya geçersiz metadata demektir; sıfır tüketim anlamına gelmez. Süre thread beklemesi ve doğrulamayı içerir.</p></section>
    <section><h2>Token kullanımı · önceki / yeni</h2>{token_comparison}<p class="muted">Yalnızca referansı değişmeyen ortak kayıtlar gösterilir. Fark yeni − önceki şeklindedir; taraflardan birinin sayacı bilinmiyorsa fark hesaplanmaz. Sağlayıcı toplamları ayrı korunur; input ve output toplanarak total üretilmez. Token sayıları maliyet tahmini veya kalite skoru değildir.</p></section>
    <section><h2>Slow drift · hareketli ortalama</h2>{drift_html}</section>
    <section><h2>Son {last_n} kayıt · pass oranı trendi</h2>{trend}<p class="muted">Birleştirilmiş raporlar tek sonuç kaydıdır; oturumları ayrı trend çalışmaları gibi gösterilmez.</p></section>
    <section><h2>Deneme geçmişi ve sınırlamalar</h2>{attempts}<p>Veriler kaydedilmiş JSON sonuçlarından okunmuştur. Bu HTML oluşturulurken API çağrısı yapılmadı. Eksik kayıtlar veya bilinmeyen token değerleri tahmin edilmedi.</p></section>
    <footer>Offline HTML · Harici script, font veya grafik servisi gerektirmez.</footer></main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--previous-dataset", type=Path)
    parser.add_argument("--last-n", type=int, default=10)
    args = parser.parse_args()
    if bool(args.previous) != bool(args.previous_dataset):
        parser.error("--previous and --previous-dataset must be supplied together")
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig")) if path else None
    html = render_report(read(args.report), read(args.dataset), previous=read(args.previous),
                         previous_dataset=read(args.previous_dataset), last_n=args.last_n)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
