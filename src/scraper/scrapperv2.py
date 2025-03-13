import os
import json
import time
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class MinecraftWikiScraper:
    BASE_URL = "https://minecraft.wiki/w/"

    # Pages that contain the full versions of important tables
    TABLE_PRIORITY_PAGES = {
        "Achievements": "Achievements",
        "Advancements": "Advancements",
        "Enchantments": "Enchanting"
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
            "tables": [],
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

        print(f"🚀 Fetching: {self.url} using Selenium...")
        driver.get(self.url)
        time.sleep(5)  # Allow page to fully load
        page_source = driver.page_source
        driver.quit()

        print("✅ Successfully fetched page with Selenium.")
        return BeautifulSoup(page_source, "html.parser")

    def extract_tables(self, soup):
        """
        Extracts tables from the page and stores them as structured JSON.
        Avoids redundant tables if a full version exists elsewhere.
        """
        content = soup.find("div", {"class": "mw-parser-output"})
        tables = content.find_all("table", {"class": "wikitable"})  # Extract only 'wikitable' tables

        current_section = None  # To track which section the table belongs to

        for element in content.find_all(["h2", "h3", "table"]):
            if element.name in ["h2", "h3"]:  # Track section headings
                current_section = element.get_text(strip=True).replace("[edit | edit source]", "")

            if element.name == "table":
                # Skip redundant tables if the full version exists on another page
                if current_section in self.TABLE_PRIORITY_PAGES and self.TABLE_PRIORITY_PAGES[current_section] != self.topic:
                    print(f"⚠️ Skipping {current_section} table on {self.topic}, full table exists on {self.TABLE_PRIORITY_PAGES[current_section]}")
                    continue

                headers = [th.get_text(strip=True) for th in element.find_all("th")]
                rows = []

                for row in element.find_all("tr")[1:]:  # Skip the header row
                    columns = [td.get_text(strip=True) for td in row.find_all("td")]
                    if columns:
                        rows.append(dict(zip(headers, columns)))  # Create row as dictionary

                if headers and rows:
                    self.data["tables"].append({
                        "section": current_section,  # Associate table with its heading
                        "headers": headers,
                        "rows": rows
                    })

    def parse_sections(self, soup):
        """
        Extracts structured sections from the page.
        """
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            raise Exception("Could not find main content div")

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
        print(f"✅ Data saved to {filename}")

    def run(self):
        """
        Runs the complete scraping pipeline.
        """
        print(f"🔍 Scraping: {self.url}")
        soup = self.fetch_page_with_selenium()
        self.parse_sections(soup)
        self.extract_tables(soup)  # Extract tables while avoiding redundancy
        self.save_to_json()


# Example usage:
if __name__ == "__main__":
    topics = ["Nether_Portal", "Diamond", "Enchanting"]
    for topic in topics:
        scraper = MinecraftWikiScraper(topic)
        scraper.run()
