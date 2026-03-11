#!/usr/bin/env python3
"""Run all tests and save results to reports/test_results/."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "results" / "test_reports"


def parse_junit(junit_path: Path) -> dict:
    """Parse junit.xml into structured data."""
    tree = ET.parse(junit_path)
    root = tree.getroot()
    suite = root.find("testsuite")
    if suite is None:
        return {"overall": {}, "by_module": {}}

    overall = {
        "total": int(suite.get("tests", 0)),
        "passed": int(suite.get("tests", 0))
        - int(suite.get("failures", 0))
        - int(suite.get("errors", 0)),
        "failed": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "time_sec": float(suite.get("time", 0)),
    }
    overall["status"] = "PASS" if overall["failed"] == 0 and overall["errors"] == 0 else "FAIL"

    by_module = {}
    for tc in suite.findall("testcase"):
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        time_val = float(tc.get("time", 0))
        status = "PASS"
        if tc.find("failure") is not None:
            status = "FAIL"
        elif tc.find("error") is not None:
            status = "ERROR"
        elif tc.find("skipped") is not None:
            status = "SKIP"

        module = (
            classname.replace("tests.", "").replace("test_", "").replace("ins_gps_fusion_lab.", "")
        )
        if module not in by_module:
            by_module[module] = {"tests": [], "passed": 0, "failed": 0}
        by_module[module]["tests"].append({"name": name, "status": status, "time_sec": time_val})
        if status == "PASS":
            by_module[module]["passed"] += 1
        else:
            by_module[module]["failed"] += 1

    return {"overall": overall, "by_module": by_module}


def write_summary_json(data: dict, junit_path: Path) -> None:
    """Write summary.json."""
    suite = ET.parse(junit_path).getroot().find("testsuite")
    run_info = {
        "timestamp": datetime.now().isoformat()[:19],
        "platform": suite.get("hostname", "unknown") if suite is not None else "unknown",
        "total_time_sec": data["overall"].get("time_sec", 0),
    }
    out = {
        "run_info": run_info,
        "overall": data["overall"],
        "by_module": data["by_module"],
    }
    (REPORTS_DIR / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


MODULE_DISPLAY_NAMES = {
    "gps_model": "1. Simulation — GPS Model (`test_gps_model.py`)",
    "imu_model": "2. Simulation — IMU Model (`test_imu_model.py`)",
    "kalman_filter": "3. Filters — Kalman Filter (`test_kalman_filter.py`)",
    "state_space": "4. Simulation — State Space (`test_state_space.py`)",
    "trajectory_generator": "5. Simulation — Trajectory Generator (`test_trajectory_generator.py`)",
}


def write_summary_md(data: dict) -> None:
    """Write summary.md."""
    o = data["overall"]
    lines = [
        "# Test Results Summary",
        "",
        (
            f"**Run**: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"**Status**: {o.get('status', '?')} | "
            f"**Total**: {o.get('total', 0)} tests in {o.get('time_sec', 0):.2f}s"
        ),
        "",
        "---",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Passed | {o.get('passed', 0)} |",
        f"| Failed | {o.get('failed', 0)} |",
        f"| Skipped | {o.get('skipped', 0)} |",
        f"| Total time | {o.get('time_sec', 0):.2f} s |",
        "",
        "---",
        "",
        "## By Module",
        "",
    ]
    for module, info in data["by_module"].items():
        title = MODULE_DISPLAY_NAMES.get(module, module)
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Test | Status |")
        lines.append("|------|--------|")
        for t in info["tests"]:
            lines.append(f"| {t['name']} | {t['status']} |")
        lines.append("")
        lines.append(f"**{info['passed']} passed, {info['failed']} failed**")
        lines.append("")
        lines.append("---")
        lines.append("")
    lines.append("## File Layout")
    lines.append("")
    lines.append("```")
    lines.append("results/")
    lines.append("└── test_reports/")
    lines.append("    ├── summary.md")
    lines.append("    ├── summary.json")
    lines.append("    ├── junit.xml")
    lines.append("    └── pytest_output.txt")
    lines.append("```")
    (REPORTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    junit_path = REPORTS_DIR / "junit.xml"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        f"--junitxml={junit_path}",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr

    (REPORTS_DIR / "pytest_output.txt").write_text(output, encoding="utf-8")
    print(output)

    if junit_path.exists():
        data = parse_junit(junit_path)
        write_summary_json(data, junit_path)
        write_summary_md(data)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
