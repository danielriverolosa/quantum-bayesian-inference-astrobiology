#!/usr/bin/env python3
"""
Unit & Integration Test Runner with GitHub Actions Step Summary Reporter.
Executes Chapter 2 (Classical) and Chapter 4 (Quantum) test suites.
Captures failures and generates detailed Markdown diagnostic reports.
"""

import io
import os
import sys
import time
import unittest


def run_suite(suite_name, test_dir):
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)
    
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    
    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time
    
    output = stream.getvalue()
    
    failure_details = []
    for test, err in result.failures:
        failure_details.append({
            "type": "FAILURE",
            "test": str(test),
            "traceback": err
        })
    for test, err in result.errors:
        failure_details.append({
            "type": "ERROR",
            "test": str(test),
            "traceback": err
        })
        
    return {
        "suite_name": suite_name,
        "test_dir": test_dir,
        "tests_run": result.testsRun,
        "failures_count": len(result.failures),
        "errors_count": len(result.errors),
        "duration": duration,
        "was_successful": result.wasSuccessful(),
        "failure_details": failure_details,
        "raw_output": output
    }


def main():
    print("=" * 70)
    print("🧪 EXECUTING UNIT & INTEGRATION TEST SUITES")
    print("=" * 70)

    suites_to_run = [
        ("Chapter 2 (Classical Baseline)", "src/chapter_2_classical/tests"),
        ("Chapter 4 (Quantum & ZNE Mitigation)", "src/chapter_4_quantum/tests")
    ]

    all_results = []
    overall_success = True

    for name, directory in suites_to_run:
        print(f"\n▶ Running {name} from {directory}...")
        res = run_suite(name, directory)
        all_results.append(res)
        print(res["raw_output"])
        if not res["was_successful"]:
            overall_success = False

    # Generate Markdown Summary for GITHUB_STEP_SUMMARY
    summary_md = []
    total_tests = sum(r["tests_run"] for r in all_results)
    total_failures = sum(r["failures_count"] for r in all_results)
    total_errors = sum(r["errors_count"] for r in all_results)
    total_time = sum(r["duration"] for r in all_results)

    if overall_success:
        summary_md.append("## ✅ Unit & Integration Tests: ALL PASSED\n")
        summary_md.append(f"*All **{total_tests}** unit and numerical consistency tests completed successfully in {total_time:.2f}s.*\n")
        summary_md.append("| Test Suite | Directory | Tests Run | Failures | Errors | Duration | Status |")
        summary_md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        for r in all_results:
            summary_md.append(f"| **{r['suite_name']}** | `{r['test_dir']}` | {r['tests_run']} | {r['failures_count']} | {r['errors_count']} | {r['duration']:.2f}s | ✅ **OK** |")
    else:
        summary_md.append("## ❌ Unit & Integration Tests: FAILED\n")
        summary_md.append(f"> **Blocking Issue:** Detected **{total_failures}** failure(s) and **{total_errors}** error(s). Downstream PDF compilation and release distribution have been **aborted**.\n")
        summary_md.append("| Test Suite | Directory | Tests Run | Failures | Errors | Duration | Status |")
        summary_md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        for r in all_results:
            status = "✅ **OK**" if r["was_successful"] else "❌ **FAILED**"
            summary_md.append(f"| **{r['suite_name']}** | `{r['test_dir']}` | {r['tests_run']} | {r['failures_count']} | {r['errors_count']} | {r['duration']:.2f}s | {status} |")

        summary_md.append("\n### 🚨 Diagnostic Tracebacks & Failure Motives:\n")
        for r in all_results:
            if r["failure_details"]:
                summary_md.append(f"#### Suite: {r['suite_name']}\n")
                for f in r["failure_details"]:
                    summary_md.append(f"<details><summary><b>[{f['type']}]</b> <code>{f['test']}</code></summary>\n")
                    summary_md.append("```text")
                    summary_md.append(f["traceback"].strip())
                    summary_md.append("```\n</details>\n")

    summary_text = "\n".join(summary_md) + "\n"

    # Write to GITHUB_STEP_SUMMARY if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write(summary_text)

    print("=" * 70)
    if overall_success:
        print(f"🎉 ALL {total_tests} TESTS PASSED (Total time: {total_time:.2f}s)")
        sys.exit(0)
    else:
        print(f"💥 TEST SUITE FAILED with {total_failures} failure(s) and {total_errors} error(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
