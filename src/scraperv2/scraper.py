import os
import json
import time
import logging
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MinecraftWikiScraper:
    BASE_URL = "https://minecraft.wiki/w/"
    TABLE_SCRAPING_PAGES = {
        "Achievements", "Advancements", "Enchanting", "Potion_Brewing", "Mobs", "Blocks", "Items"
    }

    def __init__(self, topic, output_folder="data/raw", max_retries=3):
        self.topic = topic
        self.url = f"{self.BASE_URL}{self.topic}"
        self.output_folder = output_folder
        self.output_path = os.path.join(output_folder, f"{self.topic}.json")
        self.max_retries = max_retries

        self.data = {
            "source": "Minecraft Wiki",
            "url": self.url,
            "title": topic,
            "sections": [],
            "tables": [] if topic in self.TABLE_SCRAPING_PAGES else None,
            "last_updated": str(datetime.now(timezone.utc)),
        }

    def fetch_page_with_selenium(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        logging.info(f"🚀 Fetching: {self.url} using Selenium...")
        driver.get(self.url)
        time.sleep(5)  # Let page fully load
        page_source = driver.page_source
        driver.quit()

        logging.info("✅ Successfully fetched page with Selenium.")
        return BeautifulSoup(page_source, "html.parser")

    def parse_sections(self, soup):
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            logging.warning(f"⚠️ No content found for {self.topic}")
            return

        intro_paragraph = ""
        for child in content.children:
            if not child.name:
                continue
            if child.name == "div" and "hatnote" in (child.get("class") or []):
                continue
            if child.name == "figure":
                continue
            if child.name == "p":
                intro_paragraph = child.get_text(separator=" ").strip()
                child.decompose()
                break

        if intro_paragraph:
            self.data["sections"].append({
                "heading": "Introduction",
                "text": intro_paragraph,
                "subsections": []
            })

        current_section = None
        for element in content.find_all(["h2", "h3", "p", "ul", "ol"]):
            if element.name == "h2":
                if current_section:
                    self.data["sections"].append(current_section)
                heading_text = element.text.strip().replace("[edit | edit source]", "")
                current_section = {
                    "heading": heading_text,
                    "text": "",
                    "subsections": []
                }

            elif element.name == "h3":
                if current_section:
                    subheading_text = element.text.strip().replace("[edit | edit source]", "")
                    current_section["subsections"].append({
                        "subheading": subheading_text,
                        "text": ""
                    })

            elif element.name in ["p", "ul", "ol"]:
                text_content = element.get_text(separator=" ").strip() + " "
                if current_section:
                    if current_section["subsections"]:
                        current_section["subsections"][-1]["text"] += text_content
                    else:
                        current_section["text"] += text_content

        if current_section:
            self.data["sections"].append(current_section)

    def extract_tables(self, soup):
        if self.topic not in self.TABLE_SCRAPING_PAGES:
            return
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            return

        tables = content.find_all("table", {"class": "wikitable"})

        current_section = None
        for element in content.find_all(["h2", "h3", "table"]):
            if element.name in ["h2", "h3"]:
                heading_text = element.get_text(strip=True).replace("[edit | edit source]", "")
                current_section = heading_text
            if element.name == "table":
                table_title = element.find("caption")
                if table_title:
                    table_title = table_title.get_text(strip=True)
                else:
                    table_title = current_section or "Unknown Table"

                headers = [th.get_text(strip=True) for th in element.find_all("th")]
                rows = []
                tr_list = element.find_all("tr")
                for row in tr_list[1:]:
                    columns = [td.get_text(strip=True) for td in row.find_all("td")]
                    if columns:
                        rows.append(dict(zip(headers, columns)))

                if headers and rows:
                    self.data["tables"].append({
                        "title": table_title,
                        "headers": headers,
                        "rows": rows
                    })
                    logging.info(f"📋 Extracted table: {table_title} (Rows: {len(rows)})")

    def save_to_json(self):
        os.makedirs(self.output_folder, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        logging.info(f"✅ Data saved to {self.output_path} (Sections: {len(self.data['sections'])}, Tables: {len(self.data['tables']) if self.data['tables'] else 0})")

    def run(self):
        # ✅ Skip scraping if file already exists
        if os.path.exists(self.output_path):
            logging.info(f"⏩ Skipping {self.topic}, already scraped.")
            return

        logging.info(f"🔍 Scraping: {self.url}")
        soup = self.fetch_page_with_selenium()
        self.parse_sections(soup)
        self.extract_tables(soup)
        self.save_to_json()


def load_pages_json(json_file="data/pages.json"):
    """Loads the JSON containing pages list only (no status tracking)."""
    if not os.path.exists(json_file):
        logging.error(f"File {json_file} not found.")
        return {"pages": []}

    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Load pages from pages.json (without tracking status)
    pages_data = load_pages_json("data/pages.json")
    pages = pages_data.get("pages", [])

    for page in pages:
        try:
            scraper = MinecraftWikiScraper(page["name"])
            scraper.run()
        except Exception as e:
            logging.error(f"❌ Failed to scrape {page['name']}: {e}")

        time.sleep(2)  # Small delay for politeness

if __name__ == "__main__":
    main()
