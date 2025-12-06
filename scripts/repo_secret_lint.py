  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
#!/usr/bin/env python3
"""
Simple repo linter that fails if sensitive signing keys or patterns are added.
Exit code 0 => no issues, non-zero => found issues.

Usage:
  python3 scripts/repo_secret_lint.py [path ...]
"""
import sys
import re
from pathlib import Path

PATTERNS = [
    re.compile(r"COSIGN_PRIVATE_KEY_B64"),
    re.compile(r"COSIGN_PASSWORD"),
    re.compile(r"-----BEGIN .*PRIVATE KEY-----"),
    re.compile(r"PRIVATE_KEY_B64"),
]

def scan_file(path: Path):
    issues = []
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return issues
    for p in PATTERNS:
        if p.search(text):
            issues.append((str(path), p.pattern))
    return issues

def main(paths):
    if not paths:
        paths = ["."]
    found = []
    for p in paths:
        for f in Path(p).rglob("*"):
            if f.is_file() and f.suffix not in {".png", ".jpg", ".jpeg", ".gif", ".so", ".bin"}:
                found += scan_file(f)
    if found:
        print("Potential secret/key patterns found:")
        for fn, pattern in found:
            print(f" - {fn}: matches {pattern}")
        return 2
    print("No secret signing patterns found.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
scripts/repo_secret_lint.py
