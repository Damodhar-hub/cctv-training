"""
Download IDD-FGVD (Fine-Grained Vehicle Dataset) from IIIT Hyderabad portal.

Same authentication flow as download_idd.py — the IDD portal uses
Django session login + CSRF token for all dataset downloads.

Usage:
    python download_fgvd.py --url "$FGVD_TOKEN_URL" --output /workspace/datasets/fgvd.tar.gz
"""

import argparse
import getpass
import os
import re
import sys
import time

import requests

LOGIN_URL = "https://idd.insaan.iiit.ac.in/accounts/login/"


def parse_args():
    p = argparse.ArgumentParser(description="Download IDD-FGVD dataset with authentication")
    p.add_argument(
        "--url", required=True,
        help="FGVD download URL (the token URL from the IDD portal)",
    )
    p.add_argument(
        "--output", default="fgvd.tar.gz",
        help="Output file path (default: fgvd.tar.gz)",
    )
    return p.parse_args()


def login(session: requests.Session, email: str, password: str) -> bool:
    """Authenticate with IDD portal. Returns True on success."""

    # GET login page for CSRF token
    print("Fetching CSRF token...")
    resp = session.get(LOGIN_URL)
    resp.raise_for_status()

    csrf_token = session.cookies.get("csrftoken")
    if not csrf_token:
        match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', resp.text)
        if match:
            csrf_token = match.group(1)
        else:
            print("ERROR: Could not find CSRF token")
            return False

    # POST login
    print("Logging in...")
    login_data = {
        "csrfmiddlewaretoken": csrf_token,
        "login": email,
        "username": email,
        "email": email,
        "password": password,
    }
    headers = {"Referer": LOGIN_URL}

    resp = session.post(LOGIN_URL, data=login_data, headers=headers, allow_redirects=True)
    resp.raise_for_status()

    if "/login/" in resp.url or "/accounts/login/" in resp.url:
        print(f"ERROR: Login failed (redirected to {resp.url})")
        errors = re.findall(
            r'<(?:li|div|span)[^>]*class="[^"]*(?:error|alert)[^"]*"[^>]*>(.*?)</(?:li|div|span)>',
            resp.text, re.DOTALL,
        )
        for e in errors:
            print(f"  -> {e.strip()}")
        return False

    print(f"Login successful (redirected to: {resp.url})")
    return True


def download(session: requests.Session, url: str, output_path: str):
    """Stream-download the dataset file with progress."""
    print(f"Starting download: {url}")
    print(f"Saving to: {output_path}")

    resp = session.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    content_length = resp.headers.get("Content-Length")

    print(f"  Content-Type: {content_type}")
    print(f"  Content-Length: {content_length}")

    if "text/html" in content_type:
        print("ERROR: Server returned HTML — session may have expired or URL is wrong.")
        chunk = next(resp.iter_content(500), b"")
        print(f"  First 500 chars: {chunk[:500]}")
        sys.exit(1)

    total_bytes = int(content_length) if content_length else None
    if total_bytes and total_bytes < 1_000_000:
        print(f"WARNING: File is only {total_bytes} bytes — expected several GB")
        confirm = input("  Continue anyway? (y/N): ").strip().lower()
        if confirm != "y":
            sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    downloaded = 0
    last_pct = -1
    start_time = time.time()

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    pct = int(downloaded * 100 / total_bytes)
                    if pct >= last_pct + 5:
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        gb_done = downloaded / (1024 ** 3)
                        gb_total = total_bytes / (1024 ** 3)
                        print(f"  {pct}% — {gb_done:.1f}/{gb_total:.1f} GB — {speed:.1f} MB/s")
                        last_pct = pct
                else:
                    gb_done = downloaded / (1024 ** 3)
                    if int(gb_done * 2) > int((downloaded - len(chunk)) / (1024 ** 3) * 2):
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        print(f"  Downloaded {gb_done:.1f} GB — {speed:.1f} MB/s")

    elapsed = time.time() - start_time
    final_size = os.path.getsize(output_path)
    print(f"\nDownload complete!")
    print(f"  File: {output_path}")
    print(f"  Size: {final_size / (1024**3):.2f} GB ({final_size:,} bytes)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if final_size < 1_000_000:
        print("ERROR: Downloaded file is too small — likely not the dataset!")
        sys.exit(1)


def main():
    args = parse_args()

    print("=" * 50)
    print("IDD-FGVD Dataset Download (IIIT Hyderabad)")
    print("=" * 50)
    email = input("IIIT Portal Email: ").strip()
    password = getpass.getpass("IIIT Portal Password: ")

    if not email or not password:
        print("ERROR: Email and password are required")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    })

    if not login(session, email, password):
        sys.exit(1)

    download(session, args.url, args.output)


if __name__ == "__main__":
    main()
