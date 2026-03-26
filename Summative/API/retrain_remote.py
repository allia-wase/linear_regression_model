"""
One-time POST /retrain to your Render API (multipart file upload).

Usage:
  python retrain_remote.py https://YOUR-SERVICE.onrender.com/retrain C:\\path\\to\\training.csv

Requires: Python 3.8+ (stdlib only). CSV must match prediction.py column names, including
  Have you ever had suicidal thoughts ?   (space before ?)
At least 50 rows. Filename must end in .csv
"""
from __future__ import annotations

import os
import sys
import uuid
import urllib.error
import urllib.request


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python retrain_remote.py <RETRAIN_URL> <CSV_PATH>", file=sys.stderr)
        print("Example: python retrain_remote.py https://mindease-n866.onrender.com/retrain data.csv")
        sys.exit(2)

    url = sys.argv[1].rstrip("/")
    if not url.endswith("/retrain"):
        if "/retrain" not in url:
            url = url.rstrip("/") + "/retrain"

    path = os.path.abspath(sys.argv[2])
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    if not path.lower().endswith(".csv"):
        print("File must be a .csv", file=sys.stderr)
        sys.exit(1)

    filename = os.path.basename(path)
    with open(path, "rb") as f:
        file_bytes = f.read()

    boundary = uuid.uuid4().hex
    crlf = b"\r\n"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8")
    body += file_bytes + crlf + f"--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = resp.read().decode("utf-8", errors="replace")
            print(resp.status, out)
            if resp.status >= 400:
                sys.exit(1)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(e.code, err_body, file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
