import sys
import os
import streamlit as st

# Ensure correct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.pipeline import RAGPipeline

# Initialize the RAG pipeline
pipeline = RAGPipeline()

st.title("Minecraft Wiki Chatbot")
query = st.text_input("Ask me anything about Minecraft!")

if query:
    response = pipeline.process_query(query)  # ✅ FIXED: Now expects a single return value
    st.write("### 💬 Response:")
    st.write(response)
