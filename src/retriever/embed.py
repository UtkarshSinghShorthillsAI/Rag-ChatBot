import os
import json
import chromadb
import requests
from typing import List, Dict
from langchain_community.vectorstores import Chroma

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY. Set it in your environment variables.")

class GeminiEmbedding:
    """
    Handles embedding generation using Gemini API without requiring google-generativeai SDK.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Sends texts to Gemini API for embedding.
        """
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        embeddings = []

        for text in texts:
            payload = {
                "model": "embedding-001",
                "content": {"parts": [{"text": text}]}
            }
            response = requests.post(self.API_URL, json=payload, headers=headers, params=params)
            data = response.json()

            if "embedding" in data:
                embeddings.append(data["embedding"]["values"])  # Extract embeddings
            else:
                print(f"⚠️ Error embedding text: {data}")
                embeddings.append([])  # Append empty embedding in case of failure

        return embeddings


class EmbeddingPipeline:
    """
    Handles the full embedding process:
    - Loads chunked data
    - Generates embeddings via Gemini API
    - Stores embeddings in ChromaDB
    """

    def __init__(self, input_folder="data/chunked", db_path="data/embeddings"):
        self.input_folder = input_folder
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)  # Ensure embeddings directory exists
        self.embedding_model = GeminiEmbedding(GEMINI_API_KEY)  # Using direct API calls

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection("minecraft_embeddings")

    def load_chunks(self) -> List[Dict]:
        """
        Loads all chunked JSON files.
        """
        chunks = []
        files = [f for f in os.listdir(self.input_folder) if f.endswith(".json")]

        for file in files:
            with open(os.path.join(self.input_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks.extend(data["chunks"])  # Extract chunks

        return chunks

    def convert_table_to_text(self, table_chunk: Dict) -> str:
        """
        Converts table chunk into a text-based representation.
        Example:
        "Achievements Table: Ice Bucket Challenge - Obtain a block of Obsidian, Nether - Enter the Nether dimension."
        """
        section = table_chunk.get("section", "Unknown Section")
        rows_text = []
        headers = table_chunk["headers"]

        for row in table_chunk["rows"]:
            row_text = ", ".join(f"{headers[i]}: {value}" for i, value in enumerate(row.values()))
            rows_text.append(row_text)

        return f"{section} Table: " + " | ".join(rows_text)

    def store_embeddings(self):
        """
        Embeds chunks and stores them in ChromaDB.
        """
        print("🔍 Loading chunked data...")
        chunks = self.load_chunks()

        # Separate text chunks & table chunks
        text_chunks = [chunk for chunk in chunks if "text" in chunk]
        table_chunks = [chunk for chunk in chunks if "rows" in chunk]

        texts = [chunk["text"] for chunk in text_chunks]  # Normal text chunks
        table_texts = [self.convert_table_to_text(table) for table in table_chunks]  # Table chunks as text

        # Generate embeddings
        print("⚡ Generating embeddings...")
        embeddings = self.embedding_model.embed(texts + table_texts)

        print("📥 Storing embeddings in ChromaDB...")
        # Store text chunks
        for idx, chunk in enumerate(text_chunks):
            self.collection.add(
                ids=[f"text_chunk_{idx}"],
                embeddings=[embeddings[idx]],
                metadatas=[{"heading": chunk["heading"], "chunk_id": chunk["chunk_id"]}],
                documents=[chunk["text"]]
            )

        # Store table chunks
        for idx, chunk in enumerate(table_chunks):
            table_text = table_texts[idx]
            self.collection.add(
                ids=[f"table_chunk_{idx}"],
                embeddings=[embeddings[len(text_chunks) + idx]],
                metadatas=[{"section": chunk["section"], "chunk_id": chunk["chunk_id"]}],
                documents=[table_text]
            )

        print(f"✅ Successfully stored {len(chunks)} embeddings in ChromaDB.")

    def run(self):
        """
        Runs the full embedding pipeline.
        """
        self.store_embeddings()


# Run the embedding pipeline
if __name__ == "__main__":
    embedding_pipeline = EmbeddingPipeline()
    embedding_pipeline.run()
