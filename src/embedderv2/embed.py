import os
import json
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EmbeddingGenerator:
    def __init__(self, input_dir="data/chunks", output_dir="data/embeddings", model_name="BAAI/bge-base-en"):
        """
        Initializes the embedding generator.

        Args:
            input_dir (str): Directory containing chunked JSON files.
            output_dir (str): Directory to store the embedding JSONL files.
            model_name (str): Local embedding model (SentenceTransformer).
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name

        os.makedirs(self.output_dir, exist_ok=True)

        # Load Local Embedding Model
        logging.info(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def load_json(self, filepath):
        """Loads JSON data from a file."""
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"❌ Error loading {filepath}: {e}")
            return None

    def save_jsonl(self, filename, data):
        """Saves data to a JSONL file."""
        output_path = os.path.join(self.output_dir, filename.replace(".json", ".jsonl"))
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                for entry in data:
                    json.dump(entry, file, ensure_ascii=False)
                    file.write("\n")
            logging.info(f"✅ Embeddings saved to {output_path}")
        except Exception as e:
            logging.error(f"❌ Error saving embeddings for {filename}: {e}")

    def generate_embedding(self, text):
        """Generates an embedding for the given text using bge-base-en."""
        try:
            # Disable internal progress bar for the SentenceTransformer's encoding
            return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()
        except Exception as e:
            logging.error(f"❌ Embedding error: {e}")
            return None

    def process_file(self, filename):
        """Processes a single JSON file and generates embeddings."""
        output_filepath = os.path.join(self.output_dir, filename.replace(".json", ".jsonl"))

        # ✅ Skip if embeddings already exist
        if os.path.exists(output_filepath):
            logging.info(f"⏭️  Skipping {filename}, embeddings already exist.")
            return

        logging.info(f"🔍 Processing {filename}")
        filepath = os.path.join(self.input_dir, filename)
        chunks = self.load_json(filepath)

        if not chunks:
            logging.warning(f"⚠️ Skipped {filename} due to loading error.")
            return

        embedded_data = []
        # One progress bar per file for processing chunks
        for chunk in tqdm(chunks, desc=f"Embedding {filename}", leave=False):
            text = chunk.get("text", "").strip()
            if not text:
                continue  # Skip empty texts

            embedding = self.generate_embedding(text)
            if embedding:
                embedded_data.append({
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "title": chunk.get("title", "Untitled"),
                    "text": text,
                    "source": chunk.get("source", "Unknown"),
                    "embedding": embedding
                })

        if embedded_data:
            self.save_jsonl(filename, embedded_data)

    def run(self):
        """Runs the embedding generation for all chunked JSON files."""
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        # Use one overall progress bar for files
        for file in files:
            out_file = os.path.join(self.output_dir, file.replace(".json", ".jsonl"))
            if os.path.exists(out_file):
                logging.info(f"⏩ Skipping {file}, already embedded.")
                continue

            self.process_file(file)


if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    embedder.run()
