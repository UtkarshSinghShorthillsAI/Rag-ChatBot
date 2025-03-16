import os
import json
import logging
import chromadb
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VectorStore:
    def __init__(self, input_dir="data/embeddings", db_dir="data/vector_db", collection_name="minecraft_wiki"):
        """
        Initializes the vector store.

        Args:
            input_dir (str): Directory containing JSONL embedding files.
            db_dir (str): Directory to store the ChromaDB vector database.
            collection_name (str): Name of the ChromaDB collection.
        """
        self.input_dir = input_dir
        self.db_dir = db_dir
        self.collection_name = collection_name

        os.makedirs(self.db_dir, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def load_jsonl(self, filepath):
        """Loads JSONL data from a file."""
        data = []
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line.strip()))
            return data
        except Exception as e:
            logging.error(f"❌ Error loading {filepath}: {e}")
            return []

    def chunk_exists(self, chunk_id):
        """
        Checks if a chunk ID already exists in ChromaDB.
        
        Args:
            chunk_id (str): The ID of the chunk to check.

        Returns:
            bool: True if the chunk already exists, False otherwise.
        """
        try:
            existing_chunk = self.collection.get(ids=[chunk_id])
            return bool(existing_chunk["ids"])  # If IDs exist, chunk is already present
        except Exception as e:
            logging.error(f"⚠️ Error checking chunk existence: {e}")
            return False

    def add_to_vector_db(self, embeddings):
        """Adds embeddings to the vector database, skipping duplicates."""
        skipped = 0
        added = 0

        for entry in tqdm(embeddings, desc="📥 Adding to ChromaDB"):
            chunk_id = entry["chunk_id"]
            vector = entry["embedding"]
            document = entry["text"]
            metadata = {
                "title": entry["title"],
                "source": entry["source"],
                "text": entry["text"]
            }

            # ✅ **Check if chunk already exists before adding**
            if self.chunk_exists(chunk_id):
                logging.info(f"⏭️  Skipping {chunk_id}, already in DB.")
                skipped += 1
                continue

            # Add to ChromaDB
            self.collection.add(ids=[chunk_id], embeddings=[vector], documents=[document], metadatas=[metadata])
            added += 1

        logging.info(f"✅ Added {added} new chunks, Skipped {skipped} existing chunks.")

    def process_files(self):
        """Processes all JSONL files in the input directory."""
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".jsonl")]
        for file in files:
            filepath = os.path.join(self.input_dir, file)
            logging.info(f"🔍 Processing {file} for vector storage...")

            embeddings = self.load_jsonl(filepath)
            if embeddings:
                self.add_to_vector_db(embeddings)
                logging.info(f"✅ Processed {len(embeddings)} embeddings from {file}")

    def run(self):
        """Runs the vector store process."""
        self.process_files()
        logging.info("🎯 All embeddings stored in the vector database!")


if __name__ == "__main__":
    vector_store = VectorStore()
    vector_store.run()
