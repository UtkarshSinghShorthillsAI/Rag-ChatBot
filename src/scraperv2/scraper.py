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
        "Achievements",
        "Advancements",
        "Enchanting",
        "Potion_Brewing",
        "Mobs",
        "Blocks",
        "Items"
    }

    def __init__(self, topics, max_retries=3):
        """
        Initializes the scraper for multiple topics.
        Args:
            topics (list): List of topics to scrape.
            max_retries (int): Max retries for Selenium.
        """
        self.topics = topics
        self.max_retries = max_retries
        self.browser = self.init_browser()

    def init_browser(self):
        """
        Initializes a single Selenium browser instance.
        """
        options = Options()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def fetch_page(self, topic):
        """
        Fetches the page source using Selenium.
        """
        url = f"{self.BASE_URL}{topic}"
        logging.info(f"🚀 Fetching: {url}")

        self.browser.get(url)
        time.sleep(5)  # Allow time for the page to load
        return BeautifulSoup(self.browser.page_source, "html.parser")

    def parse_sections(self, soup, topic):
        """
        Extracts structured sections from the page.
        """
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            logging.warning(f"⚠️ No content found for {topic}")
            return []

        sections = []
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
            sections.append({"heading": "Introduction", "text": intro_paragraph, "subsections": []})

        current_section = None
        for element in content.find_all(["h2", "h3", "p", "ul", "ol"]):
            if element.name == "h2":
                if current_section:
                    sections.append(current_section)
                heading_text = element.text.strip().replace("[edit | edit source]", "")
                current_section = {"heading": heading_text, "text": "", "subsections": []}

            elif element.name == "h3":
                if current_section:
                    subheading_text = element.text.strip().replace("[edit | edit source]", "")
                    current_section["subsections"].append({"subheading": subheading_text, "text": ""})

            elif element.name in ["p", "ul", "ol"]:
                text_content = element.get_text(separator=" ").strip() + " "
                if current_section:
                    if current_section["subsections"]:
                        current_section["subsections"][-1]["text"] += text_content
                    else:
                        current_section["text"] += text_content

        if current_section:
            sections.append(current_section)

        return sections

    def extract_tables(self, soup, topic):
        """
        Extracts tables from the page.
        """
        if topic not in self.TABLE_SCRAPING_PAGES:
            return []

        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            return []

        tables = []
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
                    tables.append({"title": table_title, "headers": headers, "rows": rows})
                    logging.info(f"📋 Extracted table: {table_title} (Rows: {len(rows)})")

        return tables

    def save_to_json(self, topic, sections, tables):
        """
        Saves the extracted data to a JSON file.
        """
        os.makedirs("data/raw", exist_ok=True)
        filename = f"data/raw/{topic}.json"

        data = {
            "source": "Minecraft Wiki",
            "url": f"{self.BASE_URL}{topic}",
            "title": topic,
            "sections": sections,
            "tables": tables if topic in self.TABLE_SCRAPING_PAGES else None,
            "last_updated": str(datetime.now(timezone.utc)),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logging.info(f"✅ Data saved: {filename}")

    def run(self):
        """
        Scrapes multiple pages in one Selenium session.
        """
        for topic in self.topics:
            json_path = f"data/raw/{topic}.json"

            # Skip if already scraped
            if os.path.exists(json_path):
                logging.info(f"⏭️  Skipping {topic} (Already Scraped)")
                continue

            try:
                soup = self.fetch_page(topic)
                sections = self.parse_sections(soup, topic)
                tables = self.extract_tables(soup, topic)
                self.save_to_json(topic, sections, tables)

            except Exception as e:
                logging.error(f"❌ Error scraping {topic}: {e}")

        self.browser.quit()  # Close browser after all pages are scraped


def load_pages_json(json_file="data/pages.json"):
    """
    Loads the JSON file with page names.
    """
    if not os.path.exists(json_file):
        logging.error(f"File {json_file} not found.")
        return {"pages": []}

    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """
    Runs the batch scraper for all pages.
    """
    pages_data = load_pages_json("data/pages.json")
    pages = pages_data.get("pages", [])

    batch_size = 5  # Change this to control how many pages to scrape per batch

    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        scraper = MinecraftWikiScraper(batch)
        scraper.run()
        time.sleep(2)  # Short delay to be polite to the server


if __name__ == "__main__":
    main()
