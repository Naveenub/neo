"""Phase 1 — Matrix lore scraper. Scrapy spider, CC-BY-SA content only, robots.txt respected."""
import json
from pathlib import Path

import scrapy
from scrapy.crawler import CrawlerProcess

from neo_data.provenance import hash_file, log_provenance

MIN_BODY_CHARS = 200


def parse_lore_page(response) -> dict:
    """Extract title, body_text, categories, url from a fandom page response."""
    title = response.css("h1.page-header__title::text").get(default="").strip()
    paragraphs = response.css("div.mw-parser-output > p::text").getall()
    body_text = "\n".join(p.strip() for p in paragraphs if p.strip())
    categories = response.css("div.page-header__categories a::text").getall()
    return {
        "title": title,
        "body_text": body_text,
        "categories": [c.strip() for c in categories],
        "url": response.url,
    }


def filter_lore_doc(doc: dict) -> bool:
    """True = keep. Reject stubs and redirect pages."""
    if len(doc.get("body_text", "")) < MIN_BODY_CHARS:
        return False
    if doc.get("body_text", "").lower().startswith("redirect"):
        return False
    return True


def write_lore_jsonl(docs: list[dict], output_path: str) -> int:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    return len(docs)


class MatrixLoreSpider(scrapy.Spider):
    name = "matrix_lore"
    custom_settings = {"ROBOTSTXT_OBEY": True, "DOWNLOAD_DELAY": 1.0}

    def __init__(self, base_url: str, collected: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [base_url]
        self.collected = collected  # shared list the caller reads after crawl completes

    def parse(self, response):
        doc = parse_lore_page(response)
        if filter_lore_doc(doc):
            self.collected.append(doc)
        for href in response.css("a.category-page__member-link::attr(href)").getall():
            yield response.follow(href, callback=self.parse)


def scrape_matrix_fandom(base_url: str, output_dir: str) -> str:
    """Run the spider synchronously, filter results, write JSONL, log provenance."""
    collected: list[dict] = []
    process = CrawlerProcess(settings={"LOG_LEVEL": "WARNING"})
    process.crawl(MatrixLoreSpider, base_url=base_url, collected=collected)
    process.start()  # blocks until crawl finishes

    output_path = str(Path(output_dir) / "matrix_lore.jsonl")
    doc_count = write_lore_jsonl(collected, output_path)

    log_provenance(
        source_url=base_url,
        file_hash=hash_file(output_path),
        record_count=doc_count,
        output_path=output_path,
    )
    return output_path


if __name__ == "__main__":
    scrape_matrix_fandom("https://matrix.fandom.com/wiki/Category:Matrix_Universe", "data/raw")
