import os
import json
import google.generativeai as genai
from tqdm import tqdm

class EmbeddingGenerator:
    def __init__(self, input_dir="data/chunks", output_dir="data/embeddings", model="models/embedding-001"):
        """
        Initializes the embedding generator.

        Args:
            input_dir (str): Directory containing chunked JSON files.
            output_dir (str): Directory to store the embedding JSONL files.
            model (str): Google Gemini embedding model.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model

        os.makedirs(self.output_dir, exist_ok=True)

        # Load API key (ensure it's set in environment variables)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def load_json(self, filepath):
        """Loads JSON data from a file."""
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None

    def save_jsonl(self, filename, data):
        """Saves data to a JSONL file."""
        output_path = os.path.join(self.output_dir, filename.replace(".json", ".jsonl"))
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                for entry in data:
                    json.dump(entry, file, ensure_ascii=False)
                    file.write("\n")
            print(f"✅ Embeddings saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving embeddings for {filename}: {e}")

    def generate_embedding(self, text):
        """Generates an embedding for the given text using Gemini."""
        try:
            response = genai.embed_content(model=self.model, content=text, task_type="retrieval_document")
            return response["embedding"]
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None

    def process_file(self, filename):
        """Processes a single JSON file and generates embeddings."""
        print(f"🔍 Processing {filename}")
        filepath = os.path.join(self.input_dir, filename)
        chunks = self.load_json(filepath)

        if not chunks:
            print(f"⚠️ Skipped {filename} due to loading error.")
            return

        embedded_data = []
        for chunk in tqdm(chunks, desc=f"Embedding {filename}"):
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "unknown")
            title = chunk.get("title", "Untitled")
            source = chunk.get("source", "Unknown")

            if not text.strip():
                continue  # Skip empty texts

            embedding = self.generate_embedding(text)
            if embedding:
                embedded_data.append({
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text,
                    "source": source,
                    "embedding": embedding
                })

        if embedded_data:
            self.save_jsonl(filename, embedded_data)

    def run(self):
        """Runs the embedding generation for all chunked JSON files."""
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        for file in files:
            self.process_file(file)


if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    embedder.run()
