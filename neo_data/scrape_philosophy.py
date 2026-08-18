"""Phase 1 — Philosophy corpus acquisition. SEP sitemap + Project Gutenberg, direct downloads."""
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

from neo_data.provenance import hash_file, log_provenance

SEP_BASE = "https://plato.stanford.edu"
SEP_SITEMAP = f"{SEP_BASE}/sitemap.xml"
GUTENBERG_SEARCH = "https://www.gutenberg.org/ebooks/search/?query="


def parse_sep_html(html: str) -> str:
    """Extract body text from an SEP entry, stripping nav/sidebar chrome."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#aueditable") or soup.select_one("#content")
    if content is None:
        return ""
    for tag in content.select("nav, .toc, script, style"):
        tag.decompose()
    return content.get_text(separator="\n", strip=True)


def download_sep_entries(output_dir: str) -> int:
    """Fetch the SEP sitemap, download each entry, extract text, write one file per entry."""
    out_dir = Path(output_dir) / "sep"
    out_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(SEP_SITEMAP) as resp:
        sitemap_xml = resp.read().decode("utf-8")
    soup = BeautifulSoup(sitemap_xml, "xml")
    entry_urls = [loc.text for loc in soup.find_all("loc") if "/entries/" in loc.text]

    count = 0
    for url in entry_urls:
        try:
            with urllib.request.urlopen(url) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
        text = parse_sep_html(html)
        if not text:
            continue
        slug = url.rstrip("/").split("/")[-1]
        (out_dir / f"{slug}.txt").write_text(text, encoding="utf-8")
        count += 1

    log_provenance(
        source_url=SEP_SITEMAP,
        file_hash=hash_file(str(out_dir / f"{entry_urls[0].rstrip('/').split('/')[-1]}.txt")) if count else "",
        record_count=count,
        output_path=str(out_dir),
    )
    return count


def download_gutenberg_texts(titles: list[str], output_dir: str) -> int:
    """Direct-download Gutenberg plaintext editions for each given book ID or slug."""
    out_dir = Path(output_dir) / "gutenberg"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for title in titles:
        # `title` is expected to be a Gutenberg ebook ID (e.g. "4363" for Kant's Critique)
        url = f"https://www.gutenberg.org/cache/epub/{title}/pg{title}.txt"
        dest = out_dir / f"{title}.txt"
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception:
            continue
        log_provenance(
            source_url=url,
            file_hash=hash_file(str(dest)),
            record_count=1,
            output_path=str(dest),
        )
        count += 1
    return count


if __name__ == "__main__":
    n_sep = download_sep_entries("data/raw")
    # Descartes, Kant, Nietzsche, Camus, Baudrillard — Gutenberg ebook IDs to be
    # confirmed per title/translation before running.
    n_gut = download_gutenberg_texts(titles=[], output_dir="data/raw")
    print(f"SEP entries: {n_sep}, Gutenberg texts: {n_gut}")
