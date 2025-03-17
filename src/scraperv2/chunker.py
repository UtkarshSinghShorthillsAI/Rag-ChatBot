import os
import json
import uuid
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Chunker:
    def __init__(
        self,
        input_dir="data/processed",
        output_dir="data/chunks",
        chunk_size=400,
        chunk_overlap=80,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.output_dir, exist_ok=True)

        # This splitter is used for sections that exceed chunk_size (if is_table=False).
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def load_json(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"❌ Error loading {filepath}: {e}")
            return None

    def save_chunks(self, filename, chunks):
        output_path = os.path.join(self.output_dir, filename)
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(chunks, file, ensure_ascii=False, indent=4)
            logging.info(f"✅ Chunks saved to {output_path}")
        except Exception as e:
            logging.error(f"❌ Error saving chunks for {filename}: {e}")

    def chunk_document(self, document, page_title=""):
        chunks = []

        for section in document:
            content = section.get("content", "")
            title = section.get("title", "Untitled")
            source = section.get("source", "Unknown")
            is_table = section.get("is_table", False)  # from preprocessor, default False

            # For crafting recipe chunks, prepend the page title to the section title.
            # Compare lowercased and stripped title for equality.
            if title.strip().lower() == "crafting recipe":
                display_title = f"{page_title} - Crafting Recipe"
            else:
                display_title = title

            # Prepend the title to content for each chunk
            if is_table:
                unique_id = uuid.uuid4().hex[:8]
                chunk_text = f"{display_title}\n\n{content}"
                chunks.append({
                    "title": display_title,
                    "chunk_id": f"{display_title.replace(' ', '_')}_{unique_id}",
                    "text": chunk_text,
                    "source": source
                })
            else:
                # Non-table section
                if len(content) <= self.chunk_size:
                    unique_id = uuid.uuid4().hex[:8]
                    chunk_text = f"{display_title}\n\n{content}"
                    chunks.append({
                        "title": display_title,
                        "chunk_id": f"{display_title.replace(' ', '_')}_{unique_id}",
                        "text": chunk_text,
                        "source": source
                    })
                else:
                    # Split with RecursiveCharacterTextSplitter
                    split_texts = self.splitter.split_text(content)
                    for sub_chunk in split_texts:
                        unique_id = uuid.uuid4().hex[:8]
                        chunk_text = f"{display_title}\n\n{sub_chunk}"
                        chunks.append({
                            "title": display_title,
                            "chunk_id": f"{display_title.replace(' ', '_')}_{unique_id}",
                            "text": chunk_text,
                            "source": source
                        })

        return chunks

    def process_file(self, filename):
        output_filepath = os.path.join(self.output_dir, filename)
        
        # ✅ Skip if chunks already exist
        if os.path.exists(output_filepath):
            logging.info(f"⏩ Skipping {filename}, already chunked.")
            return

        logging.info(f"🔍 Processing {filename}")
        filepath = os.path.join(self.input_dir, filename)
        document = self.load_json(filepath)

        if document is None:
            logging.warning(f"⚠️ Skipped {filename} due to loading error.")
            return

        # Derive page_title from filename, e.g., "Iron_Axe.json" -> "Iron Axe"
        page_title = filename.rsplit(".", 1)[0].replace("_", " ")

        chunks = self.chunk_document(document, page_title)
        self.save_chunks(filename, chunks)

    def run(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        for file in files:
            self.process_file(file)

if __name__ == "__main__":
    chunker = Chunker()
    chunker.run()
