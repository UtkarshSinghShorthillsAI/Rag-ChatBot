import os
import json
import re
from typing import List, Dict

class Chunker:
    def __init__(self, input_folder="data/processed", output_folder="data/chunked", chunk_size=500):
        """
        Initializes the chunking class.
        - `input_folder`: Where processed JSON files are stored.
        - `output_folder`: Where chunked JSON files will be saved.
        - `chunk_size`: Maximum token size per chunk.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.chunk_size = chunk_size
        os.makedirs(output_folder, exist_ok=True)

    def split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Splits a long text into multiple chunks, ensuring sentence boundaries are maintained.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)  # Split at sentence boundaries
        chunks, current_chunk = [], ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Splits large text sections into smaller chunks.
        """
        chunked_sections = []
        for section in sections:
            section_heading = section["heading"]
            section_text = section["text"]

            text_chunks = self.split_text_into_chunks(section_text, self.chunk_size)

            for idx, chunk in enumerate(text_chunks):
                chunked_sections.append({
                    "heading": section_heading,
                    "text": chunk,
                    "chunk_id": f"{section_heading}_{idx + 1}"
                })

            # Process subsections
            for subsection in section["subsections"]:
                subheading = subsection["subheading"]
                subtext = subsection["text"]
                sub_chunks = self.split_text_into_chunks(subtext, self.chunk_size)

                for idx, sub_chunk in enumerate(sub_chunks):
                    chunked_sections.append({
                        "heading": f"{section_heading} > {subheading}",
                        "text": sub_chunk,
                        "chunk_id": f"{subheading}_{idx + 1}"
                    })

        return chunked_sections

    def chunk_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Splits large tables into smaller chunks.
        """
        chunked_tables = []
        for table in tables:
            section = table.get("section", "Unknown Section")
            headers = table["headers"]
            rows = table["rows"]

            # Split tables if they exceed 10 rows
            for i in range(0, len(rows), 10):
                chunked_tables.append({
                    "section": section,
                    "headers": headers,
                    "rows": rows[i : i + 10],  # Take a batch of 10 rows
                    "chunk_id": f"{section}_TableChunk_{i // 10 + 1}"
                })

        return chunked_tables

    def process_file(self, filename):
        """
        Reads a processed JSON file, chunks its contents, and saves the chunked version.
        """
        input_path = os.path.join(self.input_folder, filename)
        output_path = os.path.join(self.output_folder, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunked_data = {
            "source": data["source"],
            "url": data["url"],
            "title": data["title"],
            "chunks": [],
            "last_updated": data["last_updated"]
        }

        # Chunk sections
        chunked_data["chunks"].extend(self.chunk_sections(data["sections"]))

        # Chunk tables
        chunked_data["chunks"].extend(self.chunk_tables(data["tables"]))

        # Save chunked data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunked_data, f, indent=4, ensure_ascii=False)

        print(f"✅ Chunked: {filename}")

    def run(self):
        """
        Runs the chunking pipeline on all processed JSON files.
        """
        files = [f for f in os.listdir(self.input_folder) if f.endswith(".json")]
        for file in files:
            self.process_file(file)


# Example usage:
if __name__ == "__main__":
    chunker = Chunker()
    chunker.run()
