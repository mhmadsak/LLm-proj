import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_SEARCH_ENGINE")

def google_search(query):
    """Perform a Google Custom Search and return first result link."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_KEY, "cx": GOOGLE_CX, "q": query}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    items = r.json().get("items", [])
    if not items:
        return None
    return items[0]["link"]

def get_page_text(url):
    """Fetch raw text from a webpage (very minimal)."""
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(" ", strip=True)[:2000]  # truncate
    except Exception:
        return None

def retrieve_context(fact):
    """Retrieve supporting context for a fact."""
    link = google_search(fact)
    if not link:
        return ""
    return get_page_text(link) or ""