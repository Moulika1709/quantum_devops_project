import json
import os
import datetime
import config


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _timestamp():
    return datetime.datetime.utcnow().isoformat() + "Z"


def generate_json_report(all_results, comparison=None):
    """
    Generates a consolidated JSON report from all algorithm results.
    all_results: dict of algorithm -> list of result records
    comparison: optional simulator vs hardware comparison dict
    """
    _ensure_dir(config.REPORTS_DIR)

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"report_{ts}.json"
    filepath = os.path.join(config.REPORTS_DIR, filename)

    summary = {}
    for algo, records in all_results.items():
        summary[algo] = {
            "total_runs": len(records),
            "backends_tested": list({r["backend"] for r in records}),
            "validations": [r.get("validation", {}) for r in records],
        }

    report = {
        "generated_at": _timestamp(),
        "framework": "DevOps Continuous Testing Framework for Quantum Algorithms",
        "config": {
            "shots": config.SHOTS,
            "chi_squared_alpha": config.CHI_SQUARED_ALPHA,
            "fidelity_threshold": config.FIDELITY_THRESHOLD,
        },
        "summary": summary,
        "simulator_vs_hardware_comparison": comparison or {},
    }

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"JSON report saved: {filepath}")
    return filepath


def _validation_badge(passed):
    if passed is True:
        return '<span class="badge pass">PASS</span>'
    elif passed is False:
        return '<span class="badge fail">FAIL</span>'
    return '<span class="badge skip">N/A</span>'


def generate_html_report(all_results, comparison=None):
    """
    Generates a human-readable HTML report for academic submission review.
    """
    _ensure_dir(config.REPORTS_DIR)

    ts_label = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    ts_file = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"report_{ts_file}.html"
    filepath = os.path.join(config.REPORTS_DIR, filename)

    rows = ""
    for algo, records in all_results.items():
        for record in records:
            v = record.get("validation", {})
            passed = v.get("passed", None)
            backend = record.get("backend", "unknown")
            shots = record.get("shots", config.SHOTS)
            ts = record.get("timestamp", "")

            detail_items = ""
            for key, val in v.items():
                if key == "passed":
                    continue
                if isinstance(val, dict):
                    val_str = json.dumps(val, indent=2)
                    detail_items += f"<li><b>{key}:</b><pre>{val_str}</pre></li>"
                else:
                    detail_items += f"<li><b>{key}:</b> {val}</li>"

            rows += f"""
            <tr>
                <td>{algo.upper()}</td>
                <td>{backend}</td>
                <td>{shots}</td>
                <td>{ts}</td>
                <td>{_validation_badge(passed)}</td>
                <td><ul>{detail_items}</ul></td>
            </tr>
            """

    comparison_section = ""
    if comparison:
        comp_rows = ""
        for algo, comp_data in comparison.items():
            fidelity = comp_data.get("fidelity", "N/A")
            passed = comp_data.get("passed", None)
            mae = comp_data.get("mean_absolute_error", "N/A")
            comp_rows += f"""
            <tr>
                <td>{algo.upper()}</td>
                <td>{fidelity}</td>
                <td>{mae}</td>
                <td>{_validation_badge(passed)}</td>
            </tr>
            """
        comparison_section = f"""
        <h2>Simulator vs Hardware Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Algorithm</th>
                    <th>Fidelity</th>
                    <th>Mean Absolute Error</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>{comp_rows}</tbody>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quantum CI/CD Test Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f4f4f4; color: #222; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; margin-bottom: 30px; }}
  th {{ background: #1a1a2e; color: #fff; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  tr:hover {{ background: #f0f0f0; }}
  pre {{ background: #eee; padding: 6px; font-size: 12px; overflow-x: auto; }}
  ul {{ margin: 0; padding-left: 18px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px; font-weight: bold; font-size: 13px; }}
  .pass {{ background: #d4edda; color: #155724; }}
  .fail {{ background: #f8d7da; color: #721c24; }}
  .skip {{ background: #e2e3e5; color: #383d41; }}
  .meta {{ font-size: 13px; color: #555; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>DevOps Continuous Testing Framework - Quantum Algorithm Report</h1>
<p class="meta">Generated: {ts_label} | Shots: {config.SHOTS}
  | Chi-squared alpha: {config.CHI_SQUARED_ALPHA} | Fidelity threshold: {config.FIDELITY_THRESHOLD}</p>

<h2>Algorithm Test Results</h2>
<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Backend</th>
      <th>Shots</th>
      <th>Timestamp</th>
      <th>Status</th>
      <th>Validation Details</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

{comparison_section}
</body>
</html>"""

    with open(filepath, "w") as f:
        f.write(html)

    print(f"HTML report saved: {filepath}")
    return filepath
