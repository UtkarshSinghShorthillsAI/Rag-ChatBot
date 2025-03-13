import os
import json
import requests
from typing import List, Dict
from src.retriever.vector_db import VectorDB  # Import our new modular DB class

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

            # Debugging API Response
            print(f"🔍 API Response ({response.status_code}): {response.text}")

            try:
                data = response.json()
                if "embedding" in data:
                    embeddings.append(data["embedding"]["values"])
                else:
                    print(f"⚠️ API returned unexpected response: {data}")
                    embeddings.append([])  # Append empty embedding in case of failure
            except requests.exceptions.JSONDecodeError:
                print(f"❌ Failed to parse JSON response. Full response: {response.text}")
                embeddings.append([])

        return embeddings


class EmbeddingPipeline:
    """
    Handles the full embedding process:
    - Loads chunked data
    - Generates embeddings via Gemini API
    - Stores embeddings in VectorDB
    """

    def __init__(self, input_folder="data/chunked"):
        self.input_folder = input_folder
        self.embedding_model = GeminiEmbedding(GEMINI_API_KEY)  # Using direct API calls
        self.vector_db = VectorDB()  # Using modular VectorDB

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

    def store_embeddings(self):
        """
        Embeds chunks and stores them in VectorDB.
        """
        print("🔍 Loading chunked data...")
        chunks = self.load_chunks()

        texts = [chunk["text"] for chunk in chunks if "text" in chunk]
        table_texts = [self.vector_db.convert_table_to_text(chunk) for chunk in chunks if "rows" in chunk]

        print("⚡ Generating embeddings...")
        embeddings = self.embedding_model.embed(texts + table_texts)

        # Store in VectorDB
        self.vector_db.add_embeddings(embeddings, chunks)

    def run(self):
        """
        Runs the full embedding pipeline.
        """
        self.store_embeddings()


# Run the embedding pipeline
if __name__ == "__main__":
    embedding_pipeline = EmbeddingPipeline()
    embedding_pipeline.run()
