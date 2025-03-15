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

    # Pages where we need to extract tables
    TABLE_SCRAPING_PAGES = {
        "Achievements",
        "Advancements",
        "Enchanting",
        "Potion_Brewing",
        "Mobs",
        "Blocks",
        "Items"
    }

    def __init__(self, topic, max_retries=3):
        """
        Initializes the scraper for a given topic.
        """
        self.topic = topic.replace(" ", "_")  # Convert spaces to underscores
        self.url = f"{self.BASE_URL}{self.topic}"
        self.data = {
            "source": "Minecraft Wiki",
            "url": self.url,
            "title": topic,
            "sections": [],
            "tables": [] if topic in self.TABLE_SCRAPING_PAGES else None,  # Only scrape tables for specific pages
            "last_updated": str(datetime.now(timezone.utc)),
        }
        self.max_retries = max_retries

    def fetch_page_with_selenium(self):
        """
        Fetches the page using Selenium (Headless Chrome).
        """
        options = Options()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        options.add_argument("--disable-blink-features=AutomationControlled")  # Bypass bot detection
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        logging.info(f"🚀 Fetching: {self.url} using Selenium...")
        driver.get(self.url)
        time.sleep(5)  # Allow page to fully load
        page_source = driver.page_source
        driver.quit()

        logging.info("✅ Successfully fetched page with Selenium.")
        return BeautifulSoup(page_source, "html.parser")

    def extract_tables(self, soup):
        """
        Extracts tables from the page ONLY for specific pages.
        """
        if self.topic not in self.TABLE_SCRAPING_PAGES:
            return  # Skip table extraction for other pages

        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            logging.warning(f"⚠️ No content found for {self.topic}")
            return

        tables = content.find_all("table", {"class": "wikitable"})

        current_section = None  # To track which section the table belongs to

        for element in content.find_all(["h2", "h3", "table"]):
            if element.name in ["h2", "h3"]:  # Track section headings
                heading_text = element.get_text(strip=True).replace("[edit | edit source]", "")
                current_section = heading_text

            if element.name == "table":
                # Get table title from <caption>, else use closest section heading
                table_title = element.find("caption")
                if table_title:
                    table_title = table_title.get_text(strip=True)
                else:
                    table_title = current_section or "Unknown Table"

                headers = [th.get_text(strip=True) for th in element.find_all("th")]
                rows = []

                # Skip the first row if it’s obviously headers
                tr_list = element.find_all("tr")
                for row in tr_list[1:]:
                    columns = [td.get_text(strip=True) for td in row.find_all("td")]
                    if columns:
                        rows.append(dict(zip(headers, columns)))  # Create row as dictionary

                if headers and rows:
                    self.data["tables"].append({
                        "title": table_title,
                        "headers": headers,
                        "rows": rows
                    })
                    logging.info(f"📋 Extracted table: {table_title} (Rows: {len(rows)})")

    def parse_sections(self, soup):
        """
        Extracts structured sections from the page, capturing:
        - Exactly one 'Introduction' paragraph after skipping any hatnotes or figures.
        - Remaining content as normal sections.
        """
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            logging.warning(f"⚠️ No content found for {self.topic}")
            return

        # 1) First, find and store exactly one real paragraph (Introduction),
        #    skipping hatnotes/figures/etc. at the top.
        intro_paragraph = ""
        for child in content.children:
            # Skip anything that isn't a Tag (e.g. NavigableString)
            if not child.name:
                continue

            # If it's a hatnote or figure, skip it
            if child.name == "div" and "hatnote" in (child.get("class") or []):
                continue
            if child.name == "figure":
                continue

            # The first <p> we see is our introduction
            if child.name == "p":
                intro_paragraph = child.get_text(separator=" ").strip()
                # Remove this <p> so we don't parse it again below
                child.decompose()
                break

        # If we found an intro paragraph, store it as "Introduction"
        if intro_paragraph:
            self.data["sections"].append({
                "heading": "Introduction",
                "text": intro_paragraph,
                "subsections": []
            })

        # 2) Now parse the rest of the sections as before
        current_section = None
        for element in content.find_all(["h2", "h3", "p", "ul", "ol"]):
            if element.name == "h2":
                # Close off any previous section
                if current_section:
                    self.data["sections"].append(current_section)

                # Start a new section
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
                    # If there's a sub-section, append text to the last sub-section
                    if current_section["subsections"]:
                        current_section["subsections"][-1]["text"] += text_content
                    else:
                        current_section["text"] += text_content

        # If there's a leftover current_section, append it
        if current_section:
            self.data["sections"].append(current_section)


    def save_to_json(self, folder="data/raw"):
        """
        Saves the extracted data to a JSON file.
        """
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{self.topic}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        table_count = len(self.data["tables"]) if self.data["tables"] else 0
        logging.info(f"✅ Data saved to {filename} (Sections: {len(self.data['sections'])}, Tables: {table_count})")

    def run(self):
        """
        Runs the complete scraping pipeline.
        """
        logging.info(f"🔍 Scraping: {self.url}")
        soup = self.fetch_page_with_selenium()
        self.parse_sections(soup)
        self.extract_tables(soup)  # Extract tables only if the page is in TABLE_SCRAPING_PAGES
        self.save_to_json()


# Example usage:
if __name__ == "__main__":
    topics = [
        "Nether_Portal", "Diamond", "Pillager", "Enchanting",
        "Achievements", "Advancements", "Bow", "Pickaxe",
        "The_Nether", "The_End", "Piglin"
    ]
    topics2 = ["Piglin"]
    for topic in topics2:
        scraper = MinecraftWikiScraper(topic)
        scraper.run()
