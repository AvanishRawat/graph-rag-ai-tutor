import os
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

from project.database.store_raw import save_raw_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "website"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (AI Tutor Crawler)"}

ALLOWED_PREFIXES = [
    "https://pantelis.github.io/courses/ai/",
    "https://pantelis.github.io/book/",
    "https://pantelis.github.io/aiml-common/",
    "https://pantelis.github.io/aiml-common/lectures/",
]

CRAWL_DELAY = 0.4

def rewrite_to_html(url: str) -> str:
    if url.endswith(".qmd"):
        return url[:-4] + "html"
    if url.endswith(".md"):
        return url[:-2] + "html"
    return url


def normalize(url: str) -> str:
    url = url.split("#")[0].strip()
    parsed = urlparse(url)

    # normalize trailing slash
    if parsed.path.endswith("/"):
        parsed = parsed._replace(path=parsed.path.rstrip("/"))

    return parsed.geturl()


def allowed(url: str) -> bool:
    for pref in ALLOWED_PREFIXES:
        if url.startswith(pref):
            return True
    return False


def get_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return None
        return r.text
    except:
        return None


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        abs_url = urljoin(base_url, a["href"])
        abs_url = normalize(abs_url)
        if allowed(abs_url):
            links.add(abs_url)

    return list(links)


def extract_pdf_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    pdfs = []

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href.lower().endswith(".pdf"):
            pdfs.append(href)

    return list(set(pdfs))


def extract_youtube_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    yt = []

    # <a> links
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if "youtube.com" in href or "youtu.be" in href:
            yt.append(href)

    # iframe embeds
    for iframe in soup.find_all("iframe", src=True):
        src = urljoin(base_url, iframe["src"])
        if "youtube.com" in src or "youtu.be" in src:
            yt.append(src)

    return list(set(yt))


def save_record(url, html, pdfs, yts, children):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else ""

    record = {
        "url": url,
        "title": title,
        "html": html,
        "pdf_links": pdfs,
        "youtube_links": yts,
        "child_links": children,
    }

    fn = f"{hash(url)}.json"
    with open(OUT_DIR / fn, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    save_raw_document(url, html)


def crawl(start_url):
    start_url = normalize(start_url)

    if not allowed(start_url):
        raise ValueError(f"Start URL '{start_url}' is outside allowed prefixes.")

    to_visit = [start_url]
    visited = set()

    print(f"Starting crawl at: {start_url}")

    while to_visit:
        url = to_visit.pop(0)
        url = normalize(url)

        if url in visited:
            continue

        visited.add(url)
        print(f"[OK] {url}")

        html = get_page(url)
        if not html:
            print(f"[FAIL] {url}")
            continue

        # resource discovery
        child_links = extract_links(html, url)
        pdf_links = extract_pdf_links(html, url)
        yt_links = extract_youtube_links(html, url)

        save_record(url, html, pdf_links, yt_links, child_links)

        for link in child_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        time.sleep(CRAWL_DELAY)

    print(f" Done. Crawled {len(visited)} pages.")


if __name__ == "__main__":
    seed = input("Enter start URL: ").strip()
    crawl(seed)
