import os
import json
import re

class Preprocessor:
    def __init__(self, input_folder="data/raw", output_folder="data/processed"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        # Define unwanted table titles/sections
        self.irrelevant_table_keywords = {"history", "navigation"}

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
        unwanted_headings = {"Gallery", "References", "Issues", "Achievements", "Sounds", "Advancements",
                             "Contents", "Navigation", "History", "See Also"}
        cleaned_sections = []

        for section in sections:
            # Use "heading" if available; if not, check for "subheading"
            heading = self.clean_text(section.get("heading", section.get("subheading", "")))
            if not heading or any(uw.lower() in heading.lower() for uw in unwanted_headings):
                continue

            section_text = self.clean_text(section.get("text", ""))
            subsections = []
            for sub in section.get("subsections", []):
                sub_heading = self.clean_text(sub.get("subheading", ""))
                sub_text = self.clean_text(sub.get("text", ""))
                if sub_heading or sub_text:
                    subsections.append({
                        "subheading": sub_heading,
                        "text": sub_text,
                        "subsections": sub.get("subsections", [])
                    })
            if section_text.strip() or subsections:
                cleaned_sections.append({
                    "heading": heading,
                    "text": section_text,
                    "subsections": subsections
                })

        return cleaned_sections

    def flatten_sections(self, sections, parent_heading=""):
        """
        Flattens the nested sections into a list of dicts,
        each with { 'title': ..., 'content': ..., 'is_table': False }.
        """
        flattened = []
        for section in sections:
            current_heading = section.get("heading") or section.get("subheading", "")
            full_heading = (f"{parent_heading} - {current_heading}"
                            if parent_heading and current_heading
                            else current_heading or parent_heading)

            section_text = section.get("text", "")
            if section_text.strip():
                flattened.append({
                    "title": full_heading,
                    "content": section_text,
                    "is_table": False
                })

            subs = section.get("subsections", [])
            if subs:
                flattened.extend(self.flatten_sections(subs, full_heading))
        return flattened

    def clean_table(self, table, page_title=""):
        """
        Flattens a table into row-based chunks,
        each chunk will have 'is_table': True so the chunker can handle it differently.
        """
        table_title = self.clean_text(table.get("title", ""))
        section = self.clean_text(table.get("section", ""))
        if not table_title:
            table_title = section if section else f"{page_title.capitalize()} Table"

        # Filter out irrelevant tables based on keywords (e.g., History, Navigation)
        if any(kw in table_title.lower() for kw in self.irrelevant_table_keywords) or \
           any(kw in section.lower() for kw in self.irrelevant_table_keywords):
            return []

        headers = [self.clean_text(h) for h in table.get("headers", []) if h.strip()]
        rows = table.get("rows", [])
        flattened_rows = []

        for row in rows:
            if headers:
                row_values = []
                for header in headers:
                    value = self.clean_text(row.get(header, ""))
                    if value:
                        row_values.append(f"{header}: {value}")
                if not row_values:
                    row_values = [
                        f"{self.clean_text(k)}: {self.clean_text(v)}"
                        for k, v in row.items() if self.clean_text(v)
                    ]
            else:
                row_values = [
                    f"{self.clean_text(k)}: {self.clean_text(v)}"
                    for k, v in row.items() if self.clean_text(v)
                ]

            if row_values:
                row_content = "; ".join(row_values)
                flattened_rows.append({
                    "title": table_title,
                    "content": row_content,
                    "is_table": True
                })
        return flattened_rows

    def simplify_crafting_recipe(self, recipe):
        """
        Simplifies the crafting_recipe field into a consistent chunk.
        The ingredients are cleaned and the grid_cleaned (if available) is formatted.
        The formatted grid shows each row on a new line with cells enclosed in square brackets.
        """
        ingredients = self.clean_text(recipe.get("ingredients", ""))
        grid_cleaned = recipe.get("grid_cleaned")
        grid_text = ""
        if isinstance(grid_cleaned, list):
            # Convert each row (list) into a string, each cell within square brackets
            grid_text = "\n".join([" ".join([f"[ {cell} ]" if cell else "[    ]" for cell in row])
                                   for row in grid_cleaned])
        elif isinstance(recipe.get("grid_raw"), str):
            grid_text = self.clean_text(recipe.get("grid_raw"))
        
        content = f"Ingredients: {ingredients}\nCrafting Grid:\n{grid_text}"
        return {
            "title": "Crafting Recipe",
            "content": content,
            "is_table": False
        }

    def preprocess_file(self, filename):
        print(f"🔍 Processing {filename}")
        data = self.load_json(filename)

        page_title = self.clean_text(data.get("title", ""))
        source_url = self.clean_text(data.get("url", ""))

        flattened_data = []

        # Flatten normal text sections
        if "sections" in data:
            cleaned_sections = self.filter_sections(data["sections"])
            flattened_data.extend(self.flatten_sections(cleaned_sections))

        # Flatten table data
        if "tables" in data and isinstance(data["tables"], list):
            for table in data["tables"]:
                flattened_data.extend(self.clean_table(table, page_title))

        # Process crafting_recipe as a separate chunk if present
        if "crafting_recipe" in data and data["crafting_recipe"]:
            recipe_chunk = self.simplify_crafting_recipe(data["crafting_recipe"])
            flattened_data.append(recipe_chunk)

        # Add source info to each chunk
        for chunk in flattened_data:
            chunk["source"] = source_url if source_url else page_title

        self.save_json(filename, flattened_data)

    def run(self):
        files = [f for f in os.listdir(self.input_folder) if f.endswith(".json")]
        for file in files:
            self.preprocess_file(file)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
