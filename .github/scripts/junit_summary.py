import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

ILLEGAL = re.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\ufffe\\uffff]")
ANNOTATION_CAP = 10
PASS, FAIL, SKIP, NOTRUN = range(4)


def junit_files(path):
    files = []
    for match in glob.glob(path, recursive=True):
        if os.path.isdir(match):
            files.extend(glob.glob(os.path.join(match, "**", "*.xml"), recursive=True))
        elif os.path.isfile(match):
            files.append(match)
    return sorted(set(files))


def outcomes(files):
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                root = ET.fromstring(ILLEGAL.sub("", fh.read()))
        except ET.ParseError as e:
            print(f"::warning::cannot parse {path}: {e}", file=sys.stderr)
            continue
        for case in root.iter("testcase"):
            status = (case.get("status") or "").lower()
            name = case.get("name") or "?"
            cls = case.get("classname")
            full = f"{cls}.{name}" if cls and cls != name else name
            if case.find("failure") is not None or case.find("error") is not None or status in ("fail", "failed"):
                yield FAIL, full
            elif status == "notrun":
                yield NOTRUN, full
            elif case.find("skipped") is not None or status in ("disabled", "skipped"):
                yield SKIP, full
            else:
                yield PASS, full


def annotate(path, label):
    failures = [name for outcome, name in outcomes(junit_files(path)) if outcome == FAIL]
    for name in failures[:ANNOTATION_CAP]:
        print(f"::error title={label} failed::{name}")


def row_key(artifact):
    suite, _, variant = artifact.partition("-test-results-")
    return "-".join(variant.split("-")[:2]), suite


def summary(root, needs):
    counts = defaultdict(lambda: [0, 0, 0, 0])
    failures = defaultdict(list)
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        key = row_key(name)
        tally = counts[key]
        for outcome, test in outcomes(junit_files(path)):
            tally[outcome] += 1
            if outcome == FAIL:
                failures[key].append(test)
    lines = [
        "## Test results",
        "",
        "| Platform | Suite | Passed | Failed | Skipped | Not run | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in sorted(counts):
        passed, failed, skipped, notrun = counts[key]
        note = "" if any(counts[key]) else "no results"
        lines.append(f"| {key[0]} | {key[1]} | {passed} | {failed} | {skipped} | {notrun} | {note} |")
    for job, state in sorted(needs.items()):
        if state["result"] != "success":
            lines.append(f"| - | {job} | - | - | - | - | job {state['result']} |")
    for key in sorted(failures):
        lines += ["", f"### {key[0]} {key[1]}: {len(failures[key])} failed", ""]
        lines += [f"- `{test}`" for test in failures[key]]
    print("\n".join(lines))


def main(argv):
    if argv[1:2] == ["annotate"]:
        annotate(argv[2], argv[3])
    elif argv[1:2] == ["summary"]:
        summary(argv[2], json.loads(os.environ["NEEDS"]))
    else:
        sys.exit(f"usage: {argv[0]} annotate <junit> <label> | summary <dir>")


if __name__ == "__main__":
    main(sys.argv)
