import os
import json
import re

class Preprocessor:
    def __init__(self, input_folder="data/raw", output_folder="data/processed"):
        """
        Initializes the preprocessor with input (raw) and output (cleaned) data paths.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def clean_text(self, text):
        """
        Cleans text by:
        - Removing wiki artifacts like '[edit | edit source]', '[hide]'
        - Removing Wikipedia-style references ('Jump up to:', 'See also:')
        - Normalizing whitespace and newlines
        - Keeping important Minecraft-related metadata (like '[Java Edition]')
        """
        text = re.sub(r"\[edit \| edit source\]", "", text)  # Remove '[edit | edit source]'
        text = re.sub(r"\[hide\]", "", text)  # Remove '[hide]'
        text = re.sub(r"Jump up to: .*", "", text)  # Remove 'Jump up to:' wiki references
        text = re.sub(r"See also: .*", "", text)  # Remove 'See also:' references
        text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
        return text

    def clean_heading(self, heading):
        """
        Cleans headings by removing '[edit | edit source]'.
        """
        return self.clean_text(heading)

    def clean_table(self, table):
        """
        Cleans table headers and removes unnecessary artifacts.
        Also removes empty rows and invalid headers.
        """
        cleaned_headers = [self.clean_text(header) for header in table["headers"]]
        cleaned_rows = [
            {self.clean_text(k): self.clean_text(v) for k, v in row.items() if v.strip()}
            for row in table["rows"]
        ]
        # Remove tables that have only empty or meaningless headers
        if not cleaned_headers or not cleaned_rows:
            return None
        return {"headers": cleaned_headers, "rows": cleaned_rows}

    def process_file(self, filename):
        """
        Cleans a single JSON file and saves the processed data.
        """
        input_path = os.path.join(self.input_folder, filename)
        output_path = os.path.join(self.output_folder, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Remove unwanted sections (Navigation, References, Issues, and Contents)
        unwanted_sections = {"Navigation", "References", "Issues", "Contents"}
        data["sections"] = [sec for sec in data["sections"] if sec["heading"] not in unwanted_sections]

        # Clean headings and text
        for section in data["sections"]:
            section["heading"] = self.clean_heading(section["heading"])
            section["text"] = self.clean_text(section["text"])

            # Clean subsections
            section["subsections"] = [
                {"subheading": self.clean_heading(sub["subheading"]), "text": self.clean_text(sub["text"])}
                for sub in section["subsections"]
                if sub["text"].strip()
            ]

        # Remove empty sections after processing
        data["sections"] = [sec for sec in data["sections"] if sec["text"].strip() or sec["subsections"]]

        # Clean tables and remove invalid ones
        data["tables"] = [t for t in (self.clean_table(table) for table in data["tables"]) if t]

        # Trim overly large history tables (keeping only relevant information)
        for section in data["sections"]:
            if "History" in section["heading"]:
                section["text"] = "\n".join(section["text"].split("\n")[:5])  # Keep only the first 5 lines

        # Save cleaned JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ Processed: {filename}")

    def run(self):
        """
        Runs the preprocessing pipeline on all raw JSON files.
        """
        files = [f for f in os.listdir(self.input_folder) if f.endswith(".json")]
        for file in files:
            self.process_file(file)


# Example usage:
if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
