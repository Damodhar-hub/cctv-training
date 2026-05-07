"""
Download IDD Detection dataset from IIIT Hyderabad portal.

The IDD portal requires Django session login + CSRF token.
Plain wget/curl gets redirected to the login page.

Usage:
    python download_idd.py --url "$IDD_TOKEN_URL" --output /workspace/datasets/idd_detection.tar.gz
"""

import argparse
import getpass
import os
import sys
import time

import requests

LOGIN_URL = 'https://idd.insaan.iiit.ac.in/accounts/login/'


def parse_args():
    parser = argparse.ArgumentParser(description='Download IDD dataset with authentication')
    parser.add_argument(
        '--url', required=True,
        help='IDD download URL (the token URL from the portal)',
    )
    parser.add_argument(
        '--output', default='idd_detection.tar.gz',
        help='Output file path (default: idd_detection.tar.gz)',
    )
    return parser.parse_args()


def login(session: requests.Session, email: str, password: str) -> bool:
    """Authenticate with IDD portal. Returns True on success."""

    # Step 1: GET login page to obtain CSRF token
    print("Fetching CSRF token...")
    resp = session.get(LOGIN_URL)
    resp.raise_for_status()

    csrf_token = session.cookies.get('csrftoken')
    if not csrf_token:
        # Try extracting from page HTML as fallback
        import re
        match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', resp.text)
        if match:
            csrf_token = match.group(1)
        else:
            print("ERROR: Could not find CSRF token")
            return False

    # Debug: find all input fields in the login form
    import re
    input_fields = re.findall(r'<input[^>]*name=["\']([^"\']+)["\'][^>]*>', resp.text)
    print(f"  Login form fields: {input_fields}")

    # Step 2: POST login credentials
    # Try multiple field name combinations (Django allauth vs custom)
    print("Logging in...")
    login_data = {
        'csrfmiddlewaretoken': csrf_token,
        'login': email,
        'username': email,
        'email': email,
        'password': password,
    }
    headers = {
        'Referer': LOGIN_URL,
    }

    resp = session.post(LOGIN_URL, data=login_data, headers=headers, allow_redirects=True)
    resp.raise_for_status()

    print(f"  Response URL: {resp.url}")
    print(f"  Status: {resp.status_code}")
    print(f"  Cookies: {dict(session.cookies)}")

    # Check if we got redirected back to login (= failed)
    if '/login/' in resp.url or '/accounts/login/' in resp.url:
        print(f"ERROR: Login failed. Response URL: {resp.url}")
        # Show error messages from page
        errors = re.findall(r'<(?:li|div|span)[^>]*class="[^"]*(?:error|alert)[^"]*"[^>]*>(.*?)</(?:li|div|span)>', resp.text, re.DOTALL)
        if errors:
            for e in errors:
                print(f"  -> {e.strip()}")
        elif 'incorrect' in resp.text.lower() or 'invalid' in resp.text.lower():
            print("  -> Invalid email or password")
        return False

    print(f"Login successful (redirected to: {resp.url})")
    return True


def download(session: requests.Session, url: str, output_path: str):
    """Stream-download the dataset file with progress reporting."""

    print(f"Starting download from: {url}")
    print(f"Saving to: {output_path}")

    resp = session.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()

    # Validate response is actually binary data, not HTML login page
    content_type = resp.headers.get('Content-Type', '')
    content_length = resp.headers.get('Content-Length')

    print(f"  Content-Type: {content_type}")
    print(f"  Content-Length: {content_length}")

    if 'text/html' in content_type:
        print("ERROR: Server returned HTML instead of binary data.")
        print("  This usually means the session expired or the URL is wrong.")
        print("  First 500 chars of response:")
        # Read a small chunk to show what we got
        chunk = next(resp.iter_content(500), b'')
        print(f"  {chunk[:500]}")
        sys.exit(1)

    if content_length:
        total_bytes = int(content_length)
        if total_bytes < 1_000_000:  # Less than 1MB is suspicious
            print(f"WARNING: File is only {total_bytes} bytes — expected ~22GB")
            print("  This might be an error page, not the dataset.")
            confirm = input("  Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                sys.exit(1)
    else:
        total_bytes = None
        print("  No Content-Length header — cannot show exact progress")

    # Stream download
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    downloaded = 0
    last_pct = -1
    start_time = time.time()
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                # Progress reporting
                if total_bytes:
                    pct = int(downloaded * 100 / total_bytes)
                    # Report every 5%
                    if pct >= last_pct + 5:
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        gb_done = downloaded / (1024 ** 3)
                        gb_total = total_bytes / (1024 ** 3)
                        print(f"  {pct}% — {gb_done:.1f}/{gb_total:.1f} GB — {speed:.1f} MB/s")
                        last_pct = pct
                else:
                    # No total size known, report every 500MB
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

    # Final validation
    if final_size < 1_000_000:
        print("ERROR: Downloaded file is too small — likely not the dataset!")
        sys.exit(1)


def main():
    args = parse_args()

    # Get credentials
    print("=" * 50)
    print("IDD Dataset Download (IIIT Hyderabad)")
    print("=" * 50)
    email = input("IIIT Portal Email: ").strip()
    password = getpass.getpass("IIIT Portal Password: ")

    if not email or not password:
        print("ERROR: Email and password are required")
        sys.exit(1)

    # Create session and login
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    })

    if not login(session, email, password):
        sys.exit(1)

    # Download
    download(session, args.url, args.output)


if __name__ == '__main__':
    main()
