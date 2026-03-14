import json
from project.ingest.crawler import crawl
from project.ingest.cleaning import run_cleaning
from project.ingest.pdf_parser import ingest_pdf
from project.ingest.youtube import ingest_youtube_video
from pymongo import MongoClient
import os

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("RAW_DB", "eng_ai_rag")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

RAW_PDF_COLL = "raw_pdfs"
RAW_YT_COLL = "raw_youtube"


def run_pipeline(start_url):
    print(" Starting full ingestion pipeline...\n")

    print(" Crawling website...")
    crawl_out = crawl(start_url)

    pdf_links = crawl_out["pdf_links"]
    yt_links = crawl_out["youtube_links"]

    print(f" Crawled pages: {len(crawl_out['pages'])}")
    print(f" PDF links found: {len(pdf_links)}")
    print(f" YouTube links found: {len(yt_links)}\n")

    print(" Running HTML cleaner...")
    run_cleaning()
    print("\n")

    print(" Fetching PDF content...")
    for pdf_url in pdf_links:
        rec = ingest_pdf(pdf_url)
        if rec:
            db[RAW_PDF_COLL].update_one({"url": pdf_url}, {"$set": rec}, upsert=True)

    print(f" PDFs ingested: {len(pdf_links)}\n")

    print(" Fetching YouTube transcripts...")
    for yt_url in yt_links:
        rec = ingest_youtube_video(yt_url)
        if rec:
            db[RAW_YT_COLL].update_one({"url": yt_url}, {"$set": rec}, upsert=True)

    print(f" YouTube transcripts ingested: {len(yt_links)}\n")

    print(" Full ingestion pipeline completed.")

    return {
        "pages": crawl_out["pages"],
        "pdfs": pdf_links,
        "youtube": yt_links
    }


if __name__ == "__main__":
    run_pipeline("https://pantelis.github.io/courses/ai/")
