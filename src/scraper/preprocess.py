import os
import json
import re

class Preprocessor:
    def __init__(self, input_folder="data/raw", output_folder="data/processed"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def load_json(self, filename):
        try:
            with open(os.path.join(self.input_folder, filename), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return {}

    def save_json(self, filename, data):
        try:
            with open(os.path.join(self.output_folder, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"✅ Processed: {filename}")
        except IOError as e:
            print(f"❌ Error saving {filename}: {e}")

    def clean_text(self, text):
        if not text:
            return ""

        patterns = [
            r"\[edit\s*\|\s*edit source\]",
            r"\[hide\]",
            r"Jump up to:.*",
            r"See also:.*",
            r"\[.*?\]",
            r"↑",
            r"\s+"
        ]

        for pattern in patterns[:-1]:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(patterns[-1], " ", text).strip()

        return text

    def filter_sections(self, sections):
        unwanted_headings = {"Gallery", "References", "Navigation", "Contents", "Issues",
                             "Achievements", "Sounds", "History", "Advancements"}
        cleaned_sections = []

        for section in sections:
            heading = self.clean_text(section.get("heading", ""))
            if not heading or any(uw.lower() in heading.lower() for uw in unwanted_headings):
                continue

            section_text = self.clean_text(section.get("text", ""))
            subsections = []
            for sub in section.get("subsections", []):
                subheading = self.clean_text(sub.get("subheading", ""))
                subtext = self.clean_text(sub.get("text", ""))
                if subheading or subtext:
                    subsections.append({"subheading": subheading, "text": subtext})

            if section_text.strip() or subsections:
                cleaned_sections.append({
                    "heading": heading,
                    "text": section_text,
                    "subsections": subsections
                })

        return cleaned_sections

    def flatten_sections(self, sections, parent_heading=""):
        flattened = []
        for section in sections:
            heading = section.get("heading", "")
            full_heading = f"{parent_heading} - {heading}" if parent_heading else heading

            section_text = section.get("text", "")
            if section_text:
                flattened.append({"title": full_heading, "content": section_text})

            subsections = section.get("subsections", [])
            flattened.extend(self.flatten_sections(subsections, full_heading))

        return flattened

    def clean_table(self, table, page_title=""):
        section = self.clean_text(table.get("section", ""))
        headers = [self.clean_text(h) for h in table.get("headers", []) if h.strip()]
        rows = [
            {self.clean_text(k): self.clean_text(v) for k, v in row.items() if v.strip()}
            for row in table.get("rows", []) if any(v.strip() for v in row.values())
        ]

        if not headers or not rows:
            return []

        if not section:
            section = f"{page_title.capitalize()} Table"

        flattened_rows = []
        for row in rows:
            row_content = "; ".join(f"{key}: {value}" for key, value in row.items())
            flattened_rows.append({"title": section, "content": row_content})

        return flattened_rows

    def preprocess_file(self, filename):
        print(f"🔍 Processing {filename}")
        data = self.load_json(filename)

        page_title = data.get("title", "").strip()
        source_url = data.get("url", "").strip()  # Extract source URL here

        flattened_data = []

        if "sections" in data:
            cleaned_sections = self.filter_sections(data["sections"])
            flattened_data.extend(self.flatten_sections(cleaned_sections))

        if "tables" in data and isinstance(data["tables"], list):
            for table in data["tables"]:
                flattened_data.extend(self.clean_table(table, page_title))

        # Add source URL to each chunk
        for chunk in flattened_data:
            chunk["source"] = source_url

        self.save_json(filename, flattened_data)

    def run(self):
        files = [f for f in os.listdir(self.input_folder) if f.endswith(".json")]
        for file in files:
            self.preprocess_file(file)


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
