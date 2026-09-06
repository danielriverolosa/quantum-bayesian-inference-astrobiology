#!/usr/bin/env python3
"""
LaTeX Integrity & Pre-Flight Syntax Auditor
Verifies file inclusions, brace balance, label cross-references, and citation keys.
Generates rich markdown reports for GitHub Step Summary.
"""

import os
import re
import sys
import time

TEX_DIR = "thesis"
BIB_FILE = os.path.join(TEX_DIR, "references.bib")
TARGET_ROOTS = ["main.tex", "main_digital.tex", "cover.tex"]


def audit_target(root_file):
    errors = []
    root_path = os.path.normpath(os.path.join(TEX_DIR, root_file))
    if not os.path.exists(root_path):
        return {
            "target": root_file,
            "files_count": 0,
            "labels_count": 0,
            "refs_count": 0,
            "cites_count": 0,
            "errors": [{"target": root_file, "category": "Missing Root", "file": root_file, "line": 0, "details": f"Root file not found at {root_path}"}]
        }

    # 1. Discover all included files recursively
    included_files = [root_path]
    visited = set()
    queue = [root_path]

    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)

        try:
            with open(curr, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            errors.append({
                "target": root_file,
                "category": "File Read Error",
                "file": curr,
                "line": 0,
                "details": str(e)
            })
            continue

        clean_lines = [re.sub(r"(?<!\\)%.*", "", line) for line in content.splitlines()]
        clean_text = "\n".join(clean_lines)

        incs = re.findall(r"\\(?:input|include)\{([^}]+)\}", clean_text)
        for inc in incs:
            inc_tex = inc if inc.endswith(".tex") else inc + ".tex"
            inc_path = os.path.normpath(os.path.join(TEX_DIR, inc_tex))
            if not os.path.exists(inc_path):
                errors.append({
                    "target": root_file,
                    "category": "Missing File Inclusion",
                    "file": curr,
                    "line": 0,
                    "details": f"Referenced file '{inc}' not found at '{inc_path}'"
                })
            else:
                included_files.append(inc_path)
                queue.append(inc_path)

    # 2. Check bibliography keys
    bib_keys = set()
    if os.path.exists(BIB_FILE):
        with open(BIB_FILE, "r", encoding="utf-8") as f:
            bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", f.read()))
    else:
        errors.append({
            "target": root_file,
            "category": "Missing Bib File",
            "file": BIB_FILE,
            "line": 0,
            "details": f"Bibliography file '{BIB_FILE}' not found."
        })

    # 3. Check syntax, braces, labels, refs, pagerefs, citations
    labels = {}
    refs = []
    pagerefs = []
    citations = []

    unique_files = sorted(set(included_files))
    for tf in unique_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            continue

        clean = re.sub(r"\\\{|\\\}", "", "".join(lines))
        clean_lines = [re.sub(r"(?<!\\)%.*", "", l) for l in clean.splitlines()]
        clean_text = "\n".join(clean_lines)

        o = clean_text.count("{")
        c = clean_text.count("}")
        if o != c:
            errors.append({
                "target": root_file,
                "category": "Brace Mismatch",
                "file": tf,
                "line": 0,
                "details": f"{o} open braces vs {c} close braces (difference: {o - c})"
            })

        for idx, line in enumerate(lines, 1):
            l_clean = re.sub(r"(?<!\\)%.*", "", line)

            for lbl in re.findall(r"\\label\{([^}]+)\}", l_clean):
                if lbl in labels:
                    errors.append({
                        "target": root_file,
                        "category": "Duplicate Label",
                        "file": tf,
                        "line": idx,
                        "details": f"Label '{lbl}' already defined in {labels[lbl]}"
                    })
                else:
                    labels[lbl] = f"{tf}:{idx}"

            for r in re.findall(r"\\ref\{([^}]+)\}", l_clean):
                refs.append((r, tf, idx))

            for pr in re.findall(r"\\pageref\{([^}]+)\}", l_clean):
                pagerefs.append((pr, tf, idx))

            for cites in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", l_clean):
                for ckey in cites.split(","):
                    ckey = ckey.strip()
                    if ckey:
                        citations.append((ckey, tf, idx))

    # Validate citations
    for ckey, tf, idx in citations:
        if ckey not in bib_keys:
            errors.append({
                "target": root_file,
                "category": "Missing Citation Key",
                "file": tf,
                "line": idx,
                "details": f"Citation key '\\cite{{{ckey}}}' not defined in {BIB_FILE}"
            })

    # Validate refs
    for r, tf, idx in refs:
        if r not in labels:
            errors.append({
                "target": root_file,
                "category": "Dangling \\ref",
                "file": tf,
                "line": idx,
                "details": f"Referenced label '\\ref{{{r}}}' is undefined"
            })

    # Validate pagerefs
    for pr, tf, idx in pagerefs:
        if pr not in labels:
            errors.append({
                "target": root_file,
                "category": "Dangling \\pageref",
                "file": tf,
                "line": idx,
                "details": f"Referenced page label '\\pageref{{{pr}}}' is undefined"
            })

    return {
        "target": root_file,
        "files_count": len(unique_files),
        "labels_count": len(labels),
        "refs_count": len(refs),
        "cites_count": len(citations),
        "errors": errors
    }


def main():
    start_time = time.time()
    results = []
    all_errors = []

    print("=" * 70)
    print("🔍 RUNNING LATEX INTEGRITY & CITATION PRE-FLIGHT AUDIT")
    print("=" * 70)

    for target in TARGET_ROOTS:
        res = audit_target(target)
        results.append(res)
        all_errors.extend(res["errors"])
        status = "PASSED" if not res["errors"] else f"FAILED ({len(res['errors'])} errors)"
        print(f"[{status}] Target: {target:<18} Files: {res['files_count']:<2} | Labels: {res['labels_count']:<3} | Refs: {res['refs_count']:<3} | Cites: {res['cites_count']:<3}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Audit completed in {elapsed:.3f}s. Total Errors: {len(all_errors)}")

    # Format Markdown Report for GITHUB_STEP_SUMMARY
    summary_md = []
    if not all_errors:
        summary_md.append("## ✅ LaTeX Integrity & Citation Pre-Flight Audit: PASSED\n")
        summary_md.append(f"*Verified in {elapsed:.3f} seconds with 0 syntax or cross-referencing anomalies.*\n")
        summary_md.append("| Compilation Target | Included Files | Declared Labels | Cross-References | Citations | Status |")
        summary_md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for res in results:
            summary_md.append(f"| `{res['target']}` | {res['files_count']} | {res['labels_count']} | {res['refs_count']} | {res['cites_count']} | ✅ **100% Valid** |")
    else:
        summary_md.append("## ❌ LaTeX Integrity & Citation Pre-Flight Audit: FAILED\n")
        summary_md.append(f"> **Blocking Issue:** Found **{len(all_errors)}** critical LaTeX syntax or reference error(s). Downstream PDF compilation and release distribution have been **aborted** to prevent generating damaged documents.\n")
        summary_md.append("### 🚨 Breakdown of Detected Errors:\n")
        summary_md.append("| Target | Category | File & Line | Cause & Identifier |")
        summary_md.append("| :--- | :--- | :--- | :--- |")
        for err in all_errors:
            loc = f"`{err['file']}`" + (f":{err['line']}" if err['line'] else "")
            summary_md.append(f"| `{err['target']}` | **{err['category']}** | {loc} | {err['details']} |")

    summary_text = "\n".join(summary_md) + "\n"

    # Write to GITHUB_STEP_SUMMARY if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write(summary_text)

    if all_errors:
        print("\n❌ CRITICAL ERRORS DETECTED:")
        for err in all_errors:
            print(f"  [{err['target']}] {err['category']}: {err['details']} ({err['file']}:{err['line']})")
        sys.exit(1)
    else:
        print("\n✅ All compilation targets passed full integrity and syntax verification.")
        sys.exit(0)


if __name__ == "__main__":
    main()
