import os
import json
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
            print(f"❌ Error loading {filepath}: {e}")
            return None

    def save_chunks(self, filename, chunks):
        output_path = os.path.join(self.output_dir, filename)
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(chunks, file, ensure_ascii=False, indent=4)
            print(f"✅ Chunks saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving chunks for {filename}: {e}")

    def chunk_document(self, document):
        chunks = []

        for section in document:
            content = section.get("content", "")
            title = section.get("title", "Untitled")
            source = section.get("source", "Unknown")
            is_table = section.get("is_table", False)  # from preprocessor, default false

            # We want to prepend the title to the content for each chunk
            # so that the chunk text = title + \n\n + actual text.
            # If is_table: keep as single chunk
            if is_table:
                unique_id = uuid.uuid4().hex[:8]
                # Prepend title to content
                chunk_text = f"{title}\n\n{content}"
                chunks.append({
                    "title": title,
                    "chunk_id": f"{title.replace(' ', '_')}_{unique_id}",
                    "text": chunk_text,
                    "source": source
                })
            else:
                # Non-table section
                if len(content) <= self.chunk_size:
                    # Fits in one chunk
                    unique_id = uuid.uuid4().hex[:8]
                    chunk_text = f"{title}\n\n{content}"
                    chunks.append({
                        "title": title,
                        "chunk_id": f"{title.replace(' ', '_')}_{unique_id}",
                        "text": chunk_text,
                        "source": source
                    })
                else:
                    # Split with RecursiveCharacterTextSplitter
                    split_texts = self.splitter.split_text(content)
                    for chunk in split_texts:
                        unique_id = uuid.uuid4().hex[:8]
                        # Prepend the same title to each sub-chunk
                        chunk_text = f"{title}\n\n{chunk}"
                        chunks.append({
                            "title": title,
                            "chunk_id": f"{title.replace(' ', '_')}_{unique_id}",
                            "text": chunk_text,
                            "source": source
                        })

        return chunks

    def process_file(self, filename):
        print(f"🔍 Processing {filename}")
        filepath = os.path.join(self.input_dir, filename)
        document = self.load_json(filepath)

        if document is None:
            print(f"⚠️ Skipped {filename} due to loading error.")
            return

        chunks = self.chunk_document(document)
        self.save_chunks(filename, chunks)

    def run(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        for file in files:
            self.process_file(file)

if __name__ == "__main__":
    chunker = Chunker()
    chunker.run()
