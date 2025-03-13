import os
import chromadb
from langchain_community.vectorstores import Chroma
from typing import List, Dict

class VectorDB:
    """
    Manages vector storage using ChromaDB (pluggable design for future DBs like FAISS, Pinecone).
    """

    def __init__(self, db_path="data/embeddings"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection("minecraft_embeddings")

    def add_embeddings(self, embeddings: List[List[float]], chunks: List[Dict]):
        """
        Adds embeddings and metadata to the vector database.
        """
        print("📥 Storing embeddings in VectorDB...")
        
        for idx, chunk in enumerate(chunks):
            metadata = {"heading": chunk.get("heading", ""), "chunk_id": chunk.get("chunk_id", "")}
            text = chunk.get("text", "") if "text" in chunk else self.convert_table_to_text(chunk)

            self.collection.add(
                ids=[f"chunk_{idx}"],
                embeddings=[embeddings[idx]],
                metadatas=[metadata],
                documents=[text]
            )

        print(f"✅ Successfully stored {len(chunks)} embeddings in VectorDB.")

    def convert_table_to_text(self, table_chunk: Dict) -> str:
        """
        Converts table chunk into a text-based representation for embedding.
        """
        section = table_chunk.get("section", "Unknown Section")
        rows_text = []
        headers = table_chunk["headers"]

        for row in table_chunk["rows"]:
            row_text = ", ".join(f"{headers[i]}: {value}" for i, value in enumerate(row.values()))
            rows_text.append(row_text)

        return f"{section} Table: " + " | ".join(rows_text)
