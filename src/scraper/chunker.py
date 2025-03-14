import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(
        self,
        input_dir="data/processed",
        output_dir="data/chunks",
        chunk_size=400,
        chunk_overlap=75,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.output_dir, exist_ok=True)

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
            with open(output_path := os.path.join(self.output_dir, filename), "w", encoding="utf-8") as file:
                json.dump(chunks, file, ensure_ascii=False, indent=4)
            print(f"✅ Chunks saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving chunks for {filename}: {e}")

    def chunk_document(self, document):
        chunks = []
        for section in document:
            content = section.get("content", "")
            title = section.get("title", "untitled")

            split_texts = self.splitter.split_text(content)

            for idx, chunk in enumerate(split_texts):
                chunks.append({
                    "title": section.get("title", "Untitled"),
                    "chunk_id": f"{section.get('title', 'Untitled').replace(' ', '_')}_{idx+1}",
                    "text": chunk
                })
        return chunks

    def process_file(self, filename):
        print(f"🔍 Processing {filename}")
        filepath = os.path.join(self.input_dir, filename)
        document = self.load_json(filepath)

        if document is None:
            print(f"⚠️ Skipped {filename} due to loading error")
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