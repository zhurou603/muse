"""MUSE 预训练实时看板（自包含 HTML，无外网依赖，浏览器打开即自动刷新）。

读取 run 目录的 config.resolved.yaml / train.jsonl / checkpoints/，
结合 hy-smi 快照与已跑评估结果，生成 <run_dir>/dashboard.html。

用法：
  python -m muse_encoder.pretrain.report --run-dir model_design/runs/formal-s1a
  python -m muse_encoder.pretrain.report  # 自动扫描 model_design/runs/formal-*
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import subprocess
from pathlib import Path

import yaml

RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"

_LOSS_COLORS = {
    "total": "#e11d48", "mlm": "#2563eb", "descriptor": "#059669",
    "denoise": "#d97706", "energy": "#7c3aed", "force": "#0891b2",
    "stress": "#be185d", "aux": "#65a30d", "align": "#9333ea",
    "coordinate": "#0d9488", "distance": "#ca8a04",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _smi_snapshot() -> list[dict]:
    try:
        out = subprocess.run(["hy-smi"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in out.splitlines():
        parts = line.split()
        if (len(parts) >= 7 and parts[0].isdigit() and parts[1].endswith("C")):
            try:
                cards.append({"card": int(parts[0]),
                              "vram_pct": float(parts[5].rstrip("%")),
                              "util_pct": float(parts[6].rstrip("%"))})
            except ValueError:
                continue
    return cards


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _svg_line(series: list[tuple[float, float]], width: int = 560, height: int = 160,
              color: str = "#2563eb", log_scale: bool = False, label: str = "") -> str:
    """极简纯 SVG 折线（series = [(x, y)]）。"""
    import math
    points = [(x, math.log10(y) if log_scale and y > 0 else y) for x, y in series
              if y is not None and (not log_scale or y > 0)]
    if len(points) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 <= y0:
        y1 = y0 + 1e-12
    pad = 28
    coords = " ".join(
        f"{pad + (x - x0) / max(1e-12, x1 - x0) * (width - 2 * pad):.1f},"
        f"{height - pad - (y - y0) / (y1 - y0) * (height - 2 * pad):.1f}"
        for x, y in points
    )
    ymin_label = f"1e{y0:.0f}" if log_scale else f"{y0:.3g}"
    ymax_label = f"1e{y1:.0f}" if log_scale else f"{y1:.3g}"
    return (
        f'<svg width="{width}" height="{height}" role="img" aria-label="{html.escape(label)}">'
        f'<text x="4" y="{pad - 8}" font-size="11" fill="#64748b">{html.escape(label)}'
        f'  [{x0:.0f}→{x1:.0f}]</text>'
        f'<text x="4" y="{pad + 2}" font-size="10" fill="#94a3b8">{ymax_label}</text>'
        f'<text x="4" y="{height - pad + 10}" font-size="10" fill="#94a3b8">{ymin_label}</text>'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{coords}"/>'
        f"</svg>"
    )


def _card(title: str, value: str, sub: str = "", color: str = "#0f172a") -> str:
    return (
        f'<div class="card"><div class="card-title">{html.escape(title)}</div>'
        f'<div class="card-value" style="color:{color}">{html.escape(value)}</div>'
        f'<div class="card-sub">{html.escape(sub)}</div></div>'
    )


def render_dashboard(run_dir: Path, smi: list[dict] | None = None) -> str:
    run_dir = Path(run_dir)
    config = {}
    cfg_path = run_dir / "config.resolved.yaml"
    if cfg_path.exists():
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    rows = _read_jsonl(run_dir / "train.jsonl")
    training = config.get("training", {})
    total_steps = int(training.get("total_steps", 0))
    latest = rows[-1] if rows else {}
    step = int(latest.get("step", 0))
    losses = latest.get("losses", {})
    elapsed = float(latest.get("elapsed_seconds", 0.0))
    per_step = elapsed / step if step else 0.0
    eta = _fmt_duration(per_step * (total_steps - step)) if step and total_steps else "—"
    samples = int(latest.get("samples", 0))
    world = 8  # 由 batch_samples/micro_batch 推断不出 world；固定写 8 卡约定
    samples_per_s = (samples / elapsed * world) if elapsed else 0.0

    loss_names = sorted({key for row in rows for key in row.get("losses", {})})
    charts = []
    if "total" in loss_names:
        charts.append(_svg_line([(r["step"], r["losses"]["total"]) for r in rows],
                                color=_LOSS_COLORS["total"], log_scale=True,
                                label="total loss (log10)"))
    for name in loss_names:
        if name == "total":
            continue
        charts.append(_svg_line([(r["step"], r["losses"].get(name)) for r in rows],
                                color=_LOSS_COLORS.get(name, "#334155"), log_scale=True,
                                label=f"{name} (log10)"))
    charts.append(_svg_line([(r["step"], r.get("lr", 0.0)) for r in rows],
                            color="#7c3aed", label="lr"))
    if any("grad_norm" in r for r in rows):
        charts.append(_svg_line([(r["step"], r.get("grad_norm")) for r in rows
                                 if r.get("grad_norm") is not None],
                                color="#be123c", log_scale=True, label="grad_norm (log10)"))

    mole_rows = [r for r in rows if r.get("mole")]
    mole_html = ""
    if mole_rows:
        latest_mole = mole_rows[-1]["mole"]
        lines = []
        for layer, stats in sorted(latest_mole.items()):
            entropy = stats.get("entropy")
            lines.append(f"<tr><td>{html.escape(layer)}</td><td>{entropy:.3f}</td></tr>")
        mole_html = ("<h2>MoLE 门控熵（越接近 ln(专家数)=2.08 越均衡）</h2>"
                     "<table><tr><th>layer</th><th>entropy</th></tr>"
                     + "".join(lines) + "</table>")

    checkpoints = []
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        checkpoints = sorted(p.name for p in ckpt_dir.glob("*.pt"))
    ckpt_html = "".join(f"<li>{html.escape(name)}</li>" for name in checkpoints)

    cards = _smi_snapshot() if smi is None else smi
    gpu_html = ""
    if cards:
        cells = "".join(
            f"<td class={'warn' if c['vram_pct'] < 5 or c['vram_pct'] > 97 else ''}>"
            f"{c['card']}: {c['vram_pct']:.0f}% / {c['util_pct']:.0f}%</td>"
            for c in cards
        )
        gpu_html = ("<h2>GPU（显存% / 利用率%）</h2>"
                    f"<table><tr>{cells}</tr></table>")

    evals_html = ""
    evals_path = run_dir / "evals.jsonl"
    if evals_path.exists():
        evals = _read_jsonl(evals_path)
        items = "".join(
            f"<tr><td>step {e.get('checkpoint_step')}</td><td>{html.escape(str(e.get('summary', '')))}</td></tr>"
            for e in evals[-8:]
        )
        evals_html = f"<h2>Checkpoint 探针评估</h2><table>{items}</table>"

    cards_html = "".join([
        _card("step", f"{step} / {total_steps}", f"eta {eta}"),
        _card("total loss", f"{losses.get('total', float('nan')):.4f}", color="#e11d48"),
        _card("lr", f"{latest.get('lr', 0.0):.2e}"),
        _card("grad_norm", (f"{latest['grad_norm']:.3f}" if latest.get("grad_norm") is not None else "—")),
        _card("samples/s", f"{samples_per_s:.1f}", f"{samples} samples"),
        _card("log points", str(len(rows))),
    ])
    progress = (f'<div class="bar"><div class="bar-fill" style="width:'
                f'{100.0 * step / total_steps:.1f}%"></div></div>'
                if total_steps else "")
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="120">
<title>MUSE 预训练看板 — {html.escape(run_dir.name)}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin-top: 28px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
.card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; min-width: 130px; }}
.card-title {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
.card-value {{ font-size: 22px; font-weight: 650; margin-top: 2px; }}
.card-sub {{ font-size: 11px; color: #94a3b8; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 14px; }}
.charts svg {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; }}
table {{ border-collapse: collapse; font-size: 12px; }} td, th {{ border: 1px solid #e2e8f0; padding: 3px 10px; }}
td.warn {{ background: #fef2f2; color: #b91c1c; font-weight: 700; }}
.bar {{ height: 10px; background: #e2e8f0; border-radius: 5px; overflow: hidden; margin: 8px 0 16px; }}
.bar-fill {{ height: 100%; background: #2563eb; }}
footer {{ margin-top: 28px; font-size: 11px; color: #94a3b8; }}
</style></head><body>
<h1>MUSE 预训练看板 — {html.escape(run_dir.name)}（每 2 分钟自动刷新）</h1>
{progress}
<div class="cards">{cards_html}</div>
{gpu_html}
<h2>损失曲线</h2>
<div class="charts">{''.join(charts)}</div>
{mole_html}
{evals_html}
<h2>Checkpoints</h2>
<ul>{ckpt_html}</ul>
<footer>generated {generated} · run_dir: {html.escape(str(run_dir))}</footer>
</body></html>"""


def write_dashboard(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    output = output or run_dir / "dashboard.html"
    html_text = render_dashboard(run_dir)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(html_text, encoding="utf-8")
    tmp.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="缺省时自动扫描 model_design/runs/formal-*")
    args = parser.parse_args()
    if args.run_dir is not None:
        print(write_dashboard(args.run_dir))
        return
    if not RUNS_ROOT.exists():
        raise SystemExit(f"runs 目录不存在: {RUNS_ROOT}")
    smi = _smi_snapshot()
    for run_dir in sorted(RUNS_ROOT.glob("formal-*")):
        if (run_dir / "config.resolved.yaml").exists():
            print(write_dashboard(run_dir))


if __name__ == "__main__":
    main()
