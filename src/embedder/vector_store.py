import os
import json
import chromadb
from tqdm import tqdm

class VectorStore:
    def __init__(self, input_dir="data/embeddings", db_dir="data/vector_db"):
        """
        Initializes the vector store.

        Args:
            input_dir (str): Directory containing JSONL embedding files.
            db_dir (str): Directory to store the ChromaDB vector database.
        """
        self.input_dir = input_dir
        self.db_dir = db_dir

        os.makedirs(self.db_dir, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(name="minecraft_wiki")

    def load_jsonl(self, filepath):
        """Loads JSONL data from a file."""
        data = []
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line.strip()))
            return data
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return []

    def add_to_vector_db(self, embeddings):
        """Adds embeddings to the vector database."""
        for entry in tqdm(embeddings, desc="Adding to ChromaDB"):
            chunk_id = entry["chunk_id"]
            vector = entry["embedding"]
            document = entry["text"]
            # print(f"📌 Chunk ID: {chunk_id}")
            # print(f"🔹 Text: {document[:100]}...")  # Print first 100 chars of text
            # print(f"🔹 Source: {entry.get('source', '❌ MISSING')}")
            # print("-" * 40)

            metadata = {
                "title": entry["title"],
                "source": entry["source"],
                "text": entry["text"]
            }

            # Add to ChromaDB
            self.collection.add(ids=[chunk_id], embeddings=[vector],documents=[document], metadatas=[metadata])

    def process_files(self):
        """Processes all JSONL files in the input directory."""
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".jsonl")]
        for file in files:
            filepath = os.path.join(self.input_dir, file)
            print(f"🔍 Processing {file} for vector storage...")

            embeddings = self.load_jsonl(filepath)
            if embeddings:
                self.add_to_vector_db(embeddings)
                print(f"✅ Stored {len(embeddings)} embeddings from {file}")

    def run(self):
        """Runs the vector store process."""
        self.process_files()
        print("🎯 All embeddings stored in the vector database!")


if __name__ == "__main__":
    vector_store = VectorStore()
    vector_store.run()
