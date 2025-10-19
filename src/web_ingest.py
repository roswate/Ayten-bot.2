# -*- coding: utf-8 -*-
"""
Kullanım:
  python3 src/web_ingest.py https://gaziantepagzi.com/ --max-pages 60 --depth 2
"""
import sys
import argparse
from urllib.parse import urljoin, urlparse
import time
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import trafilatura

OUT_DIR = Path(__file__).resolve().parents[0] / "data" / "kitaplar"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0 Safari/537.36 AytenBot/1.0",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

def same_domain(u, domain):
    try:
        return urlparse(u).netloc == domain
    except Exception:
        return False

def clean_url(u):
    u = re.sub(r"#.*$", "", u)      # anchor at
    u = re.sub(r"\?.*$", "", u)     # query at
    return u

def fetch_urls(seed, max_pages=30, depth=1, timeout=15):
    domain = urlparse(seed).netloc
    seed = clean_url(seed)
    seen = set()
    queue = [(seed, 0)]
    out = []

    while queue and len(out) < max_pages:
        url, d = queue.pop(0)
        url = clean_url(url)
        if url in seen:
            continue
        seen.add(url)

        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            ctype = r.headers.get("Content-Type", "")
            if r.status_code != 200 or ("text/html" not in ctype and "application/xhtml" not in ctype):
                continue
            out.append(url)
            if d < depth:
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    nxt = urljoin(url, a["href"])
                    nxt = clean_url(nxt)
                    if nxt.startswith(("mailto:", "tel:")):
                        continue
                    if same_domain(nxt, domain):
                        queue.append((nxt, d + 1))
            time.sleep(0.4)
        except Exception:
            continue

    # Basit filtreler: yönetim, etiket, kategori gibi sayfaları istemiyorsak
    filtered = []
    skip_patterns = ("/wp-admin", "/login", "/cart", "/tag/", "/category/", "/feed")
    for u in out:
        if any(s in u for s in skip_patterns):
            continue
        filtered.append(u)

    return list(dict.fromkeys(filtered))

def extract_with_trafilatura(url, timeout=20):
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_comments=False, include_tables=False,
            target_language="tr",
            favor_precision=True
        ) or ""
        return text.strip()
    except Exception:
        return ""

def extract_with_bs4(url, timeout=20):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # Yaygın gürültü seçicilerini temizle
        for s in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
            s.extract()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""

def extract_main_text(url):
    text = extract_with_trafilatura(url)
    if len(text) < 300:  # kısa/boş ise BS4 fallback
        text = extract_with_bs4(url)
    return text

def save_corpus(urls, outfile: Path, per_page_dir: Path | None = None):
    ok_count, fail = 0, []
    chunks = []

    if per_page_dir:
        per_page_dir.mkdir(parents=True, exist_ok=True)

    for i, u in enumerate(urls, start=1):
        txt = extract_main_text(u)
        if not txt or len(txt) < 120:  # çok kısa ise “başarısız” say
            fail.append(u)
            continue

        # Her sayfa için başlık + metin
        page_block = f"[URL] {u}\n{txt}\n"
        chunks.append(page_block)

        if per_page_dir:
            # İstersen tek tek de kaydet
            fname = f"page_{i:03d}.txt"
            (per_page_dir / fname).write_text(page_block, encoding="utf-8")

        ok_count += 1

    corpus = "\n\n".join(chunks).strip()
    if not corpus:
        print("[WARN] Toplanacak metin bulunamadı. (Tüm sayfalar boş/kısa gelmiş olabilir.)")
        if fail:
            print("Boş/kısa gelen örnek URL:", fail[:5])
        return False

    outfile.write_text(corpus, encoding="utf-8")
    print(f"[OK] {ok_count} sayfa işlendi, çıktı: {outfile}")
    if fail:
        print(f"[INFO] İçeriği zayıf/boş sayfa: {len(fail)} adet. Örn: {fail[:3]}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", help="Başlangıç URL (örn: https://gaziantepagzi.com/)")
    ap.add_argument("--max-pages", type=int, default=30, help="Maksimum sayfa")
    ap.add_argument("--depth", type=int, default=1, help="İç link derinliği (0: sadece seed)")
    ap.add_argument("--save-each", action="store_true", help="Her sayfayı ayrı TXT kaydet")
    args = ap.parse_args()

    urls = fetch_urls(args.seed, max_pages=args.max_pages, depth=args.depth)
    if not urls:
        print("[WARN] URL bulunamadı. Seed veya erişim engeli olabilir.")
        sys.exit(0)

    domain = urlparse(args.seed).netloc.replace(":", "_")
    out_file = OUT_DIR / f"{domain}_crawl.txt"
    per_page_dir = OUT_DIR / f"{domain}_pages" if args.save_each else None

    ok = save_corpus(urls, out_file, per_page_dir=per_page_dir)
    if not ok:
        sys.exit(0)

if __name__ == "__main__":
    main()