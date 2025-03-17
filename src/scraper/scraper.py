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
        # Initialize data without crafting_recipe key by default.
        self.data = {
            "source": "Minecraft Wiki",
            "url": self.url,
            "title": topic,
            "sections": [],
            "tables": [] if topic in self.TABLE_SCRAPING_PAGES else None,
            "last_updated": str(datetime.now(timezone.utc))
        }
        self.max_retries = max_retries

    def fetch_page_with_selenium(self):
        """
        Fetches the page using Selenium (Headless Chrome) and returns both the BeautifulSoup object and the raw HTML snapshot.
        """
        options = Options()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        options.add_argument("--disable-blink-features=AutomationControlled")
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
        return BeautifulSoup(page_source, "html.parser"), page_source

    def parse_crafting_grid(self, grid_html):
        """
        Parses the raw grid HTML for a crafting recipe.
        This implementation first attempts to use the "data-minetip-title" attribute,
        and if that is not available, it falls back to the <a> tag's "title" attribute.
        Returns a list of rows, where each row is a list of ingredient names.
        """
        grid_soup = BeautifulSoup(grid_html, "html.parser")
        grid = []
        # Find all rows using the 'mcui-row' class.
        rows = grid_soup.find_all("span", class_="mcui-row")
        for row_div in rows:
            row = []
            # Each cell is contained in a span with class "invslot".
            cells = row_div.find_all("span", class_="invslot")
            for cell in cells:
                # Try first: element with "data-minetip-title"
                element = cell.find(attrs={"data-minetip-title": True})
                if element:
                    ingredient = element["data-minetip-title"]
                else:
                    # Fallback: try the <a> tag's title attribute.
                    a_tag = cell.find("a")
                    if a_tag and a_tag.get("title"):
                        ingredient = a_tag["title"]
                    else:
                        ingredient = ""
                row.append(ingredient)
            grid.append(row)
        return grid

    def extract_crafting_recipe(self, soup):
        """
        Extracts the crafting recipe table if it exists.
        Looks for a <span> with id 'Crafting' within an <h3>, then extracts the subsequent table
        whose data-description contains 'crafting recipes'.
        Uses parse_crafting_grid() to produce a cleaned, structured grid.
        Returns a dictionary with recipe details or None if not found.
        """
        crafting_span = soup.find("span", id="Crafting")
        if crafting_span:
            crafting_heading = crafting_span.find_parent("h3")
            if crafting_heading:
                table = crafting_heading.find_next_sibling("table", class_="wikitable collapsible")
                if table and "crafting recipes" in table.get("data-description", "").lower():
                    rows = table.find_all("tr")
                    if len(rows) >= 2:
                        cells = rows[1].find_all("td")
                        if len(cells) >= 2:
                            ingredients = cells[0].get_text(separator=" ").strip()
                            grid_html = str(cells[1])
                            grid_cleaned = self.parse_crafting_grid(grid_html)
                            logging.info("🔍 Crafting recipe found and extracted.")
                            return {
                                "ingredients": ingredients,
                                "grid_raw": grid_html,
                                "grid_cleaned": grid_cleaned
                            }
        logging.info("ℹ️ No crafting recipe table found on this page.")
        return None

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
            if element.name in ["h2", "h3"]:
                heading_text = element.get_text(strip=True).replace("[edit | edit source]", "")
                current_section = heading_text

            if element.name == "table":
                # Get table title from <caption>, else use the closest section heading.
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

    def parse_sections(self, soup):
        """
        Extracts structured sections from the page, capturing:
        - Exactly one 'Introduction' paragraph after skipping hatnotes/figures.
        - Remaining content as normal sections.
        """
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            logging.warning(f"⚠️ No content found for {self.topic}")
            return

        # 1) Extract the introduction paragraph.
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

        # 2) Parse remaining sections.
        current_section = None
        for element in content.find_all(["h2", "h3", "p", "ul", "ol"]):
            if element.name == "h2":
                if current_section:
                    self.data["sections"].append(current_section)
                heading_text = element.get_text(strip=True).replace("[edit | edit source]", "")
                current_section = {
                    "heading": heading_text,
                    "text": "",
                    "subsections": []
                }
            elif element.name == "h3":
                if current_section:
                    subheading_text = element.get_text(strip=True).replace("[edit | edit source]", "")
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

    def save_to_json(self, folder="data/raw"):
        """
        Saves the extracted data to a JSON file.
        """
        os.makedirs(folder, exist_ok=True)
        # Remove the crafting_recipe key if no recipe was found.
        if self.data.get("crafting_recipe") is None:
            self.data.pop("crafting_recipe", None)
        filename = f"{folder}/{self.topic}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        table_count = len(self.data["tables"]) if self.data["tables"] else 0
        logging.info(f"✅ Data saved to {filename} (Sections: {len(self.data['sections'])}, Tables: {table_count}, Crafting Recipe: {'Yes' if 'crafting_recipe' in self.data else 'No'})")

    def run(self):
        """
        Runs the complete scraping pipeline.
        """
        logging.info(f"🔍 Scraping: {self.url}")
        soup, snapshot = self.fetch_page_with_selenium()
        # Optionally, you can store the raw HTML snapshot:
        self.data["html_snapshot"] = snapshot
        self.parse_sections(soup)
        self.extract_tables(soup)
        # Attempt to extract a crafting recipe. Only add the key if found.
        recipe = self.extract_crafting_recipe(soup)
        if recipe:
            self.data["crafting_recipe"] = recipe
        self.save_to_json()

# Example usage:
if __name__ == "__main__":
    topics = [
       "Wooden_Axe"
    ]
    for topic in topics:
        scraper = MinecraftWikiScraper(topic)
        scraper.run()
