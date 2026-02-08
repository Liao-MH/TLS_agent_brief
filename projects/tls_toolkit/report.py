from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List

from .utils import ensure_dir


def render_report_md(
    out_dir: str,
    run_meta: Dict[str, Any],
    decision_trace: Dict[str, Any],
    summary: Dict[str, Any],
    qc: Dict[str, Any],
) -> str:
    """
    Generate a clean, auditable Markdown report (no external dependencies).
    """
    out_dir = ensure_dir(out_dir)
    report_dir = ensure_dir(Path(out_dir) / "artifacts" / "report")
    report_path = str(Path(report_dir) / "report.md")

    mask = summary.get("mask", {})
    tissue = summary.get("tissue", {})
    patch_plan = summary.get("patch_plan", {})
    inference = summary.get("inference", {})
    postprocess = summary.get("postprocess", {})
    vectorize = summary.get("vectorize", {})

    lines: List[str] = []

    lines.append("# TLS/GC/ImmuneCluster Segmentation Report\n")
    lines.append("## Execution Summary\n")
    lines.append(f"- Slide: `{run_meta.get('slide_id')}`")
    lines.append(f"- Run ID: `{run_meta.get('run_id')}`")
    lines.append(f"- Device: `{run_meta.get('device')}`")
    lines.append(f"- Backend: `{run_meta.get('wsi', {}).get('backend')}`")
    lines.append(f"- Mask level: `{mask.get('mask_level')}`")
    lines.append(f"- QC score: **{qc.get('qc_score')}**")
    lines.append(f"- QC flags: `{', '.join(qc.get('qc_flags', []))}`\n")
    
    # Preset / fallback policy (auditable agent policy)
    preset = run_meta.get("preset")
    preset_req = run_meta.get("preset_requested")
    fb = bool(run_meta.get("preset_fallback_enabled", False))
    mr_req = run_meta.get("max_reruns_requested", None)
    mr_eff = run_meta.get("max_reruns_effective", None)
    lines.append("## Preset policy")
    lines.append(f"- Preset (effective): `{preset}`")
    lines.append(f"- Preset (requested): `{preset_req}`")
    lines.append(f"- Fallback: **{'enabled' if fb else 'disabled'}**")
    if mr_req is not None and mr_eff is not None:
        lines.append(f"- max_reruns: requested={mr_req}, effective={mr_eff}")
    lines.append("")
    

    lines.append("## Quantitative Results\n")
    pix = mask.get("pixel_fractions", {})
    lines.append("- Pixel fractions (by label):")
    for k in sorted(pix.keys(), key=lambda x: int(x)):
        lines.append(f"  - label {k}: {pix[k]:.6f}")
    lines.append("")
    lines.append(f"- Vector objects total: {vectorize.get('objects_total', 0)}")
    lines.append(f"- Postprocess: converted TLS components: {postprocess.get('tls_components_converted', 0)}\n")

    lines.append("## Tissue & Patch Plan\n")
    lines.append(f"- Tissue fraction (thumb): {tissue.get('tissue_fraction_thumb', 0.0):.4f}")
    lines.append(f"- Tissue components: {tissue.get('tissue_components', 0)}")
    lines.append(f"- Patches: {patch_plan.get('num_patches', 0)}")
    lines.append(f"- Patch size / overlap / stride: {patch_plan.get('patch_size')}/{patch_plan.get('overlap')}/{patch_plan.get('stride')}\n")

    lines.append("## Inference & Runtime\n")
    lines.append(f"- Processed patches: {inference.get('processed', 0)}")
    lines.append(f"- Failed patches: {inference.get('failed', 0)}")
    if inference.get("first_error"):
        lines.append(f"- First error: `{inference.get('first_error')}`")
    if inference.get("first_error_traceback"):
        tb = inference.get("first_error_traceback", "")
        # keep report readable
        tb_lines = tb.splitlines()
        tb_show = "\n".join(tb_lines[:30])
        lines.append("")
        lines.append("### First error traceback (truncated)")
        lines.append("```")
        lines.append(tb_show)
        lines.append("```")
    lines.append("")

    lines.append("## Decision Trace (Agent)\n")
    steps = decision_trace.get("steps", [])
    if not steps:
        lines.append("- No agent steps recorded.\n")
    else:
        for i, s in enumerate(steps, 1):
            lines.append(f"### Step {i}: {s.get('action')}")
            lines.append(f"- Rationale: {s.get('rationale')}")
            lines.append(f"- Evidence: {s.get('evidence')}")
            lines.append("")

    lines.append("## Reproducibility\n")
    model = run_meta.get("model", {})
    lines.append(f"- mmseg config: `{model.get('config_file')}`")
    lines.append(f"- checkpoint: `{model.get('checkpoint')}`")
    lines.append(f"- checkpoint sha256: `{model.get('checkpoint_sha256')}`")
    lines.append(f"- schema_version: `{run_meta.get('schema_version')}`")
    lines.append("")

    lines.append("## Artifacts\n")
    lines.append(f"- Full mask: `{summary.get('mask', {}).get('path')}`")
    lines.append(f"- GeoJSON: `{summary.get('vectorize', {}).get('geojson_path', '')}`")
    lines.append(f"- Objects table: `{summary.get('vectorize', {}).get('objects_table_path', '')}`")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path
