#!/usr/bin/env python3
"""
Sync triton-ascend open issues with the "Issue跟踪" sheet in Google Sheets.

Features:
  1. Fetch all open issues from GitHub (including last comment content)
  2. Send issue data to Apps Script for incremental update:
     - Existing -> update columns A-J (leave K/L/M untouched)
     - New -> append a new row
  3. Write last execution time to row 2
  4. Batch sending, 10 issues per batch, with retry
  5. All timestamps use Beijing timezone (UTC+8)

Environment variables:
  GITHUB_TOKEN     - GitHub token (auto-injected by Actions as secrets.GITHUB_TOKEN)
  SHEET_ID         - Google Sheets spreadsheet ID (configure as repository secret)
  SHEET_WEBAPP_URL - Apps Script Web App URL (configure as repository secret)
"""

import os
import sys
import time
import requests
from datetime import datetime, timezone, timedelta

# ===== Timezone =====
BEIJING_TZ = timezone(timedelta(hours=8))

# ===== GitHub Config =====
REPO_OWNER = "triton-lang"
REPO_NAME = "triton-ascend"
API_BASE = "https://api.github.com"

# ===== Google Sheets Config =====
SHEET_ID = os.environ.get("SHEET_ID", "")
SHEET_NAME = "Issue跟踪"
SHEET_WEBAPP_URL = os.environ.get("SHEET_WEBAPP_URL", "")

STATUS_LABELS = {
    "triage review",
    "triaged",
    "wait feedback",
    "resolved",
    "stale",
    "duplicated",
    "invalid",
    "wontfix",
}
TYPE_LABELS = {
    "feature request",
    "rfc",
    "question",
    "documentation",
    "installation",
    "performance",
    "bug",
    "ssbuffer",
}

# ===== GitHub API Functions =====


def make_headers(token=None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "triton-ascend-sync-issues/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def requests_get(url, headers=None, params=None, max_retries=3, timeout=60):
    """GET request with retry on network timeout."""
    for attempt in range(max_retries):
        try:
            return requests.get(url, headers=headers, params=params, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Network timeout, retry in {wait}s ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def get_remaining_from_response(resp):
    """Extract remaining API call count from response headers."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        return int(remaining)
    return None


def fetch_all_open_issues(token=None):
    """Fetch all open issues from the repo (excluding PRs)."""
    headers = make_headers(token)
    issues = []
    page = 1
    remaining = None
    while True:
        url = f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/issues"
        params = {
            "state": "open",
            "per_page": 100,
            "page": page,
            "sort": "created",
            "direction": "desc",
        }
        resp = requests_get(url, headers=headers, params=params)
        if resp.status_code == 403:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
            reset_dt = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
            wait_sec = max(0, (reset_dt - datetime.now(timezone.utc)).total_seconds())
            print(f"\nAPI rate limit! Resets in {wait_sec/60:.1f} min.")
            if not token:
                print("Set GITHUB_TOKEN env var for higher limits (5000/hr).")
            if issues:
                print(f"Got {len(issues)} issues, continuing with partial data.")
                remaining = 0
                break
            sys.exit(1)
        resp.raise_for_status()
        remaining = get_remaining_from_response(resp)
        data = resp.json()
        if not data:
            break
        count_before = len(issues)
        for item in data:
            if "pull_request" not in item:
                issues.append(item)
        print(f"  Page {page}, total {len(issues)} issues ({len(issues)-count_before} new)")
        if remaining is not None:
            print(f"  API remaining: {remaining}")
        page += 1
    return issues, remaining


def fetch_last_comment(issue_number, token=None):
    """Fetch the last comment of an issue, return dict or None."""
    headers = make_headers(token)
    url = f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_number}/comments"
    params = {"per_page": 1, "sort": "created", "direction": "desc"}
    remaining = None
    try:
        resp = requests_get(url, headers=headers, params=params)
        if resp.status_code == 403:
            return None, 0
        resp.raise_for_status()
        remaining = get_remaining_from_response(resp)
        data = resp.json()
        if data:
            return data[0], remaining
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"    Warning: failed to fetch comments for issue #{issue_number}: {e}")
    return None, remaining


def parse_dt(dt_str):
    """Parse GitHub API datetime string."""
    if dt_str is None:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


# ===== Data Processing Functions =====


def categorize_labels(label_names):
    """Split GitHub labels into status label and type label (first match each)."""
    status = ""
    type_label = ""
    for name in label_names:
        normalized = name.lower().replace("-", " ")
        if not status and normalized in STATUS_LABELS:
            status = name
        elif not type_label and normalized in TYPE_LABELS:
            type_label = name
    return status, type_label


def truncate(text, max_len=200):
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def fmt_dt(dt):
    if dt is None:
        return ""
    return dt.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


def build_issue_data(issue, last_comment):
    """Build issue data (issue number + 10 column values A-J)."""
    created_at = parse_dt(issue["created_at"])
    label_names = [l["name"] for l in issue["labels"]]
    status_labels, type_labels = categorize_labels(label_names)

    last_comment_body = ""
    last_comment_time = ""
    if last_comment:
        last_comment_body = truncate(last_comment.get("body", ""))
        last_comment_time = fmt_dt(parse_dt(last_comment["created_at"]))

    return {
        "number":
        issue["number"],
        "values": [
            issue["title"],  # A: Issue Title
            issue["html_url"],  # B: Issue Link
            "否",  # C: Is Closed
            (issue.get("user") or {}).get("login", "unknown"),  # D: Author
            fmt_dt(created_at),  # E: Created Time
            "",  # F: Closed Time
            last_comment_body,  # G: Last Comment
            last_comment_time,  # H: Last Comment Time
            status_labels,  # I: Status Label
            type_labels,  # J: Type Label
        ],
    }


# ===== Google Sheets Sync =====


def sync_to_sheet(issues_data):
    """Send issue data to Apps Script in batches (10 per batch)."""
    if not SHEET_ID:
        print("ERROR: SHEET_ID not set. Configure it as a repository secret.")
        return False

    if not SHEET_WEBAPP_URL:
        print("ERROR: SHEET_WEBAPP_URL not set. Configure it as a repository secret.")
        return False

    batch_size = 10
    total_updated = 0
    total_inserted = 0
    total_failed = 0
    num_batches = (len(issues_data) + batch_size - 1) // batch_size

    exec_time = datetime.now(BEIJING_TZ).strftime("Last execution time: %Y-%m-%d %H:%M:%S")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(issues_data))
        batch = issues_data[start:end]

        payload = {
            "mode": "sync",
            "spreadsheet_id": SHEET_ID,
            "sheet_name": SHEET_NAME,
            "exec_time": exec_time,
            "issues": batch,
        }

        success = False
        for attempt in range(5):
            cache_bust_params = {"v": int(time.time() * 1000)}
            try:
                resp = requests.post(SHEET_WEBAPP_URL, json=payload, params=cache_bust_params, timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("status") == "ok":
                        u = result.get("updates", 0)
                        ins = result.get("inserts", 0)
                        total_updated += u
                        total_inserted += ins
                        print(f"  Batch {batch_idx+1}/{num_batches}: {u} updates, {ins} inserts")
                        success = True
                        break
                    else:
                        print(f"  Batch {batch_idx+1}: unexpected response (non-retryable): {resp.text[:200]}")
                        break
                elif resp.status_code >= 500 or resp.status_code == 429:
                    print(f"  Batch {batch_idx+1}: HTTP {resp.status_code} (retryable): {resp.text[:200]}")
                else:
                    print(f"  Batch {batch_idx+1}: HTTP {resp.status_code} (non-retryable): {resp.text[:200]}")
                    break
            except (requests.exceptions.RequestException, ValueError) as e:
                print(f"  Error: {e}")
            print(f"  Batch {batch_idx+1} retry ({attempt+1}/5)...")
            time.sleep(3)

        if not success:
            total_failed += 1
            print(f"  Batch {batch_idx+1} failed, skipping")

    if total_failed > 0:
        print(f"\nSync completed with {total_failed} batch(es) failed!")
        print(f"  Total rows updated: {total_updated}")
        print(f"  Total rows inserted: {total_inserted}")
        return False

    print(f"\nSync complete!")
    print(f"  Total rows updated: {total_updated}")
    print(f"  Total rows inserted: {total_inserted}")
    return True


# ===== Main =====


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        print("GITHUB_TOKEN set, authenticated mode (5000/hr)")
    else:
        print("No GITHUB_TOKEN, anonymous mode (60/hr)")

    print(f"\nFetching open issues from {REPO_OWNER}/{REPO_NAME}...")
    for attempt in range(5):
        try:
            issues, remaining = fetch_all_open_issues(token)
            break
        except Exception as e:
            if attempt < 4:
                print(f"\nFetch failed: {e}. Retrying in 10s ({attempt+2}/5)...")
                time.sleep(10)
            else:
                print(f"\nFetch failed after 5 attempts: {e}")
                sys.exit(1)
    print(f"\nFound {len(issues)} open issues on GitHub")

    if not issues:
        print("No open issues found.")
        return

    issues_data = []
    for i, issue in enumerate(issues, 1):
        number = issue["number"]
        comment_count = issue.get("comments", 0)

        last_comment = None
        if comment_count > 0 and (remaining is None or remaining >= 1):
            last_comment, new_remaining = fetch_last_comment(number, token)
            if new_remaining is not None:
                remaining = new_remaining
            time.sleep(0.5)
        elif comment_count > 0:
            print(f"  Warning: skipping comment fetch for issue #{number} (rate limit remaining: {remaining})")

        issues_data.append(build_issue_data(issue, last_comment))

        if i % 50 == 0 or i == len(issues):
            print(f"  Progress: {i}/{len(issues)}")

    print(f"\n{'='*50}")
    print(f"  Issues to sync: {len(issues_data)}")
    print(f"{'='*50}")

    if not sync_to_sheet(issues_data):
        sys.exit(1)


if __name__ == "__main__":
    main()
