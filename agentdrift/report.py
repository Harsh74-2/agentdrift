"""Standalone HTML report: drift curve (SVG), onset distribution, metric cards."""
import html
from datetime import datetime, timezone


def _svg_curve(per_length: list, w=560, h=280, pad=40) -> str:
    """Success rate vs chain length, success observed vs no-error baseline (100%)."""
    if not per_length:
        return ""
    max_len = max(d["length"] for d in per_length)
    x = lambda v: pad + (w - 2 * pad) * (v - 1) / max(1, max_len - 1)
    y = lambda v: h - pad - (h - 2 * pad) * (v / 1.0)
    pts_obs = " ".join(f"{x(d['length']):.0f},{y(d['success']):.0f}" for d in per_length)
    baseline = [1 - (d["baseline_fail"]) for d in per_length]
    pts_base = " ".join(f"{x(d['length']):.0f},{y(min(1.0, b)):.0f}"
                        for d, b in zip(per_length, baseline))
    axes = (f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#999"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#999"/>')
    grid = "".join(f'<line x1="{pad}" y1="{y(t):.0f}" x2="{w-pad}" y2="{y(t):.0f}" '
                   f'stroke="#eee"/><text x="{pad-6}" y="{y(t)+4:.0f}" text-anchor="end" '
                   f'font-size="10" fill="#999">{t:.0%}</text>' for t in (0, .25, .5, .75, 1))
    xl = "".join(f'<text x="{x(d["length"]):.0f}" y="{h-pad+14}" text-anchor="middle" '
                 f'font-size="10" fill="#666">{d["length"]}</text>' for d in per_length)
    return (f'<svg viewBox="0 0 {w} {h}" font-family="system-ui">{grid}{axes}'
            f'<polyline points="{pts_base}" fill="none" stroke="#bbb" stroke-dasharray="4 3"/>'
            f'<polyline points="{pts_obs}" fill="none" stroke="#0ea5e9" stroke-width="2"/>'
            f'<text x="{w-pad}" y="{pad-8}" text-anchor="end" font-size="11" fill="#888">'
            f'grey dashed: independent-error baseline</text>'
            f'<text x="{w/2}" y="{h-6}" text-anchor="middle" font-size="10" fill="#666">'
            f'chain length (tool calls)</text></svg>')


def render(metrics: dict, title="agentdrift report") -> str:
    e = html.escape
    rows = "".join(
        f"<tr><td>{d['length']}</td><td>{d['success']:.0%}</td>"
        f"<td>{d['onset']:.0%}</td><td>{d['baseline_fail']:.1%}</td>"
        f"<td><b>{d['amplification']:.2f}x</b></td></tr>"
        for d in metrics["per_length"])
    card = lambda label, val: (f'<div class="card"><div class="k">{e(label)}</div>'
                               f'<div class="v">{e(str(val))}</div></div>')
    onsets = metrics.get("onset_distribution") or {}
    onset_tbl = "".join(f"<tr><td>step {i}</td><td>{n}</td></tr>"
                        for i, n in sorted(onsets.items())) or "<tr><td colspan=2>none</td></tr>"
    recovery = metrics["recovery_rate"]
    recovery = f"{recovery:.0%}" if recovery is not None else "n/a (no onset errors)"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(title)}</title>
<style>
 body{{font:15px/1.6 system-ui,sans-serif;margin:0;background:#111;color:#eee}}
 .wrap{{max-width:860px;margin:0 auto;padding:32px 20px}}
 h1{{font-size:24px;letter-spacing:-.01em}} .sub{{color:#999;margin-bottom:24px}}
 .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}
 .card{{flex:1;min-width:150px;background:#1b1d1f;border:1px solid #2a2d30;border-radius:10px;padding:14px 16px}}
 .k{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#999}}
 .v{{font-size:22px;margin-top:4px;font-variant-numeric:tabular-nums}}
 svg,.box{{background:#17181a;border:1px solid #2a2d30;border-radius:10px;padding:8px;margin:18px 0}}
 table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
 td,th{{padding:6px 10px;text-align:left;border-bottom:1px solid #26282a}}
 th{{color:#999;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
</style></head><body><div class="wrap">
<h1>agentdrift</h1>
<p class="sub">{e(title)} &middot; generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<div class="cards">
{card("success rate", f"{metrics['success_rate']:.0%}")}
{card("per-step error", f"{metrics['per_step_error_rate']:.1%}")}
{card("recovery rate", recovery)}
{card("median error growth", (f"{metrics['median_error_growth']:.2f}x" if metrics.get("median_error_growth") else "n/a"))}
</div>
<div class="box">{_svg_curve(metrics["per_length"])}</div>
<table><tr><th>length</th><th>success</th><th>onset rate</th><th>independent baseline fail</th><th>amplification</th></tr>{rows}</table>
<h3 style="margin-top:28px">Onset distribution</h3>
<table style="max-width:280px">{onset_tbl}</table>
</div></body></html>"""