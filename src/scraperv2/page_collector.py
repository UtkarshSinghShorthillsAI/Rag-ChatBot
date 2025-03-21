import os
import json
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CategoryPageCollector:
    BASE_URL = "https://minecraft.wiki/w/Category:"

    def __init__(self, category, output_file="data/pages.json", max_retries=3):
        """
        Initializes the scraper to collect article links from a category page.

        Args:
            category (str): The category name to scrape.
            output_file (str): File to store extracted page names.
            max_retries (int): Max retries for handling pagination issues.
        """
        self.category = category.replace(" ", "_")  # Convert spaces to underscores
        self.url = f"{self.BASE_URL}{self.category}"
        self.output_file = output_file
        self.max_retries = max_retries
        self.pages_collected = set()  # Use a set to prevent duplicates

        # Load existing pages.json (if exists) to avoid re-scraping
        self.load_existing_pages()

    def load_existing_pages(self):
        """Loads existing scraped pages from pages.json to avoid duplicates."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    for page in existing_data.get("pages", []):
                        self.pages_collected.add(page)
                logging.info(f"📂 Loaded {len(self.pages_collected)} existing pages from {self.output_file}")
            except Exception as e:
                logging.error(f"❌ Failed to load existing pages.json: {e}")

    def save_pages(self):
        """Saves collected pages to pages.json (without duplicates)."""
        pages_data = {"pages": list(self.pages_collected)}
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(pages_data, f, indent=4, ensure_ascii=False)
            logging.info(f"✅ Saved {len(self.pages_collected)} pages to {self.output_file}")
        except IOError as e:
            logging.error(f"❌ Error saving pages.json: {e}")

    def fetch_page_with_selenium(self, url):
        """Fetches the page using Selenium and returns a BeautifulSoup object."""
        options = Options()
        options.add_argument("--headless")  # Run headless
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        logging.info(f"🚀 Fetching: {url} using Selenium...")
        driver.get(url)
        time.sleep(3)  # Allow page to fully load

        return driver

    def extract_page_links(self, driver):
        """Extracts article page links from the 'Pages in category' section."""
        try:
            # Locate the section containing article links
            page_section = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//div[@id='mw-pages']"))
            )

            # Extract all <a> links inside this section
            links = page_section.find_elements(By.XPATH, ".//a")
            for link in links:
                page_name = link.text.strip()
                if page_name and "Category:" not in page_name:  # Avoid category links
                    self.pages_collected.add(page_name)

            logging.info(f"🔗 Found {len(links)} article links on this page.")

        except Exception as e:
            logging.warning(f"⚠️ No valid 'Pages in category' section found: {e}")

    def find_next_page(self, driver):
        """Finds and returns the 'Next page' link, if available."""
        try:
            next_link = driver.find_element(By.LINK_TEXT, "next page")
            if next_link:
                return next_link.get_attribute("href")  # Get the URL of the next page
        except Exception:
            return None  # No pagination link found

    def run(self):
        """Runs the scraper to collect all pages in the category."""
        current_url = self.url
        retry_count = 0

        while current_url and retry_count < self.max_retries:
            driver = self.fetch_page_with_selenium(current_url)
            self.extract_page_links(driver)

            next_url = self.find_next_page(driver)
            driver.quit()

            if next_url:
                logging.info(f"➡️ Moving to next page: {next_url}")
                current_url = next_url
                time.sleep(2)  # Avoid aggressive requests
            else:
                logging.info("✅ No more pagination. Scraping complete.")
                break  # Stop loop if no next page

        self.save_pages()


# Example usage:
if __name__ == "__main__":
    category = "Tools"  # Example category
    collector = CategoryPageCollector(category)
    collector.run()
