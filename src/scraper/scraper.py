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
                current_section = element.get_text(strip=True).replace("[edit | edit source]", "")

            if element.name == "table":
                # Get table title from <caption>, else use closest section heading
                table_title = element.find("caption")
                table_title = table_title.get_text(strip=True) if table_title else (current_section or "Unknown Table")

                headers = [th.get_text(strip=True) for th in element.find_all("th")]
                rows = []

                for row in element.find_all("tr")[1:]:  # Skip the header row
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
        Extracts structured sections from the page.
        """
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            logging.warning(f"⚠️ No content found for {self.topic}")
            return

        current_section = None
        for element in content.find_all(["h2", "h3", "p", "ul", "ol"]):
            if element.name == "h2":
                if current_section:
                    self.data["sections"].append(current_section)  # Store previous section
                current_section = {"heading": element.text.strip(), "text": "", "subsections": []}
            elif element.name == "h3":
                if current_section:
                    current_section["subsections"].append({"subheading": element.text.strip(), "text": ""})
            elif element.name in ["p", "ul", "ol"]:
                if current_section:
                    if current_section["subsections"]:
                        current_section["subsections"][-1]["text"] += element.get_text(separator=" ").strip() + " "
                    else:
                        current_section["text"] += element.get_text(separator=" ").strip() + " "

        if current_section:  # Add last parsed section
            self.data["sections"].append(current_section)

    def save_to_json(self, folder="data/raw"):
        """
        Saves the extracted data to a JSON file.
        """
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{self.topic}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        logging.info(f"✅ Data saved to {filename} (Sections: {len(self.data['sections'])}, Tables: {len(self.data['tables']) if self.data['tables'] else 'N/A'})")

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
        "The_Nether", "The_End"
    ]
    for topic in topics:
        scraper = MinecraftWikiScraper(topic)
        scraper.run()
