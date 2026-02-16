import argparse
import inspect
import json
from pathlib import Path
from typing import Dict
import sys

try:
    import winreg  # type: ignore
except Exception:  # pragma: no cover - non-Windows
    winreg = None


AUTH_COOKIE_KEYS = (
    "session_id",
    "session_id.sig",
    "__Secure-next-auth.session-token",
)

PREFERRED_COOKIE_KEYS = (
    # Cloudflare clearance often required to fetch creator HTML pages.
    "cf_clearance",
    # CSRF/device cookies are useful signals that we got a real session.
    "a_csrf",
    "patreon_device_id",
)

def _load_browser_cookie3():
    try:
        import browser_cookie3  # type: ignore
        return browser_cookie3
    except Exception as exc:
        raise RuntimeError(
            "browser_cookie3 is required. Install with: "
            ".venv\\Scripts\\python -m pip install browser-cookie3"
        ) from exc


def _extract_from_brave(domain_name: str) -> Dict[str, str]:
    return _extract_from_browser("brave", domain_name)


def _extract_from_browser(browser_name: str, domain_name: str) -> Dict[str, str]:
    print(f"Getting cookie from: {browser_name}")
    browser_cookie3 = _load_browser_cookie3()
    fn = getattr(browser_cookie3, browser_name, None)
    if fn is None:
        raise RuntimeError(f"browser_cookie3 does not support browser '{browser_name}'.")

    sig = inspect.signature(fn)
    kwargs = {}
    if "domain_name" in sig.parameters:
        kwargs["domain_name"] = domain_name
    cookie_jar = fn(**kwargs)

    cookies: Dict[str, str] = {}
    for cookie in cookie_jar:
        if domain_name not in (cookie.domain or ""):
            continue
        if not cookie.name or cookie.value is None:
            continue
        cookies[cookie.name] = cookie.value
    return cookies


def _default_browser_name_windows() -> str:
    if winreg is None:
        return ""
    user_choice_key = (
        r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice"
    )
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, user_choice_key) as key:
            prog_id, _ = winreg.QueryValueEx(key, "ProgId")
    except Exception:
        return ""

    prog_id = str(prog_id).lower()
    mapping = {
        "bravehtml": "brave",
        "chromehtml": "chrome",
        "firefoxurl": "firefox",
        "microsoftedgehtm": "edge",
        "opera stable": "opera",
        "operastable": "opera",
        "vivaldihtm": "vivaldi",
    }
    for k, v in mapping.items():
        if k in prog_id:
            return v
    return ""


def _candidate_browsers(explicit: str = ""):
    if explicit:
        return [explicit]
    if sys.platform.startswith("win"):
        default = _default_browser_name_windows()
        if default:
            ordered = [default, "brave", "chrome", "edge", "firefox", "opera", "vivaldi"]
            seen = set()
            out = []
            for b in ordered:
                if b not in seen:
                    out.append(b)
                    seen.add(b)
            return out
    return ["brave", "chrome", "edge", "firefox", "opera", "vivaldi"]


def _cookie_header(cookies: Dict[str, str]) -> str:
    names = list(cookies.keys())
    prioritized = [name for name in AUTH_COOKIE_KEYS if name in cookies]
    rest = sorted([name for name in names if name not in AUTH_COOKIE_KEYS])
    ordered = prioritized + rest
    return "; ".join(f"{name}={cookies[name]}" for name in ordered)


def _score_cookie_set(cookies: Dict[str, str]) -> int:
    score = 0
    if any(k in cookies for k in AUTH_COOKIE_KEYS):
        score += 100
    if "cf_clearance" in cookies:
        score += 50
    for k in PREFERRED_COOKIE_KEYS:
        if k in cookies:
            score += 5
    score += min(len(cookies), 50)  # tie-breaker: more cookies is usually better
    return score


def _write_tokens_file(project_root: Path, cookie_header: str) -> Path:
    tokens_dir = project_root / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    output = tokens_dir / "patreon"
    output.write_text(cookie_header, encoding="utf-8")
    return output


def _write_global_json(project_root: Path, cookie_header: str) -> Path:
    cfg_path = project_root / "confs" / "global.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    patreon = data.get("patreon")
    if not isinstance(patreon, dict):
        patreon = {}
        data["patreon"] = patreon
    patreon["cookie"] = cookie_header
    cfg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Patreon cookies from your default browser and output as a Cookie header string."
    )
    parser.add_argument(
        "--domain",
        default="patreon.com",
        help="Cookie domain filter (default: patreon.com).",
    )
    parser.add_argument(
        "--browser",
        default="",
        help="Force a specific browser (brave/chrome/edge/firefox/opera/vivaldi). Default: auto-detect.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the cookie header only; do not write files.",
    )
    parser.add_argument(
        "--write-tokens",
        action="store_true",
        help="Write cookie header to tokens/patreon.",
    )
    parser.add_argument(
        "--write-global",
        action="store_true",
        help="Write cookie header to confs/global.json (patreon.cookie).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cookies: Dict[str, str] = {}
    selected_browser = ""
    errors = []
    best_score = -1
    best = None
    best_browser = ""

    for browser in _candidate_browsers(args.browser.strip().lower()):
        try:
            candidate = _extract_from_browser(browser, args.domain)
            if candidate:
                score = _score_cookie_set(candidate)
                if score > best_score:
                    best_score = score
                    best = candidate
                    best_browser = browser
                errors.append(
                    f"{browser}: {len(candidate)} cookies, auth={any(k in candidate for k in AUTH_COOKIE_KEYS)}, "
                    f"cf_clearance={'cf_clearance' in candidate}"
                )
            else:
                errors.append(f"{browser}: no cookies")
        except Exception as exc:
            errors.append(f"{browser}: {exc}")

    if best is not None:
        cookies = best
        selected_browser = best_browser

    if not cookies:
        raise RuntimeError(
            "No Patreon cookies found from detected browsers. "
            "Make sure Patreon is open and logged in. Tried: "
            + "; ".join(errors)
        )

    print(f"Using browser: {selected_browser}")
    cookie_header = _cookie_header(cookies)
    found_auth = [key for key in AUTH_COOKIE_KEYS if key in cookies]
    if found_auth:
        print(f"Auth cookie keys found: {', '.join(found_auth)}")
    else:
        print(
            "WARNING: No Patreon auth cookie key found "
            "(expected one of: session_id, session_id.sig, __Secure-next-auth.session-token)."
        )

    if args.print_only or (not args.write_tokens and not args.write_global):
        print(cookie_header)

    if args.write_tokens:
        out_path = _write_tokens_file(project_root, cookie_header)
        print(f"Wrote: {out_path}")

    if args.write_global:
        out_path = _write_global_json(project_root, cookie_header)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
