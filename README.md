# Minecraft-ChatBot

## **Introduction**  

### **Why a Minecraft RAG Bot?**  
Did you know that a **fully generated Minecraft world** is **over 30 million times larger than Earth**? With **thousands of mechanics, blocks, mobs, and recipes**, even veteran players can’t remember everything. Constantly searching the **Minecraft Wiki** can be tedious—so why not let AI do it for you?  

This **Minecraft RAG Bot** makes information retrieval effortless. Using **Retrieval-Augmented Generation (RAG)**, it provides **instant, accurate answers** to any Minecraft-related query. Just **ask in natural language**, and get precise, up-to-date responses without digging through pages.  

### **What This Documentation Covers**  
This documentation is structured into the following sections:  

1. **Project Overview and Architecture** – Goals and technology stack.  
2. **Directory Structure** – Breakdown of files and modules.   
3. **Scraper** – Extracting and structuring wiki data.  
4. **Embedding** – Storing game knowledge for efficient retrieval.  
5. **Pipeline** – How queries are processed to generate responses.  
6. **Frontend** – A lightweight Streamlit UI for easy access.  


## **Project Overview**  

This **Minecraft RAG Bot** is designed to provide **quick and accurate answers** to **Minecraft-related queries** by leveraging **Retrieval-Augmented Generation (RAG)**. Instead of manually searching the **Minecraft Wiki**, the bot allows users to **ask questions in natural language** and receive **contextually relevant responses**.  

### **How It Works**  
1. **Scraper** – Extracts structured data from **Minecraft Wiki** (one-time process).  
2. **Embedding** – Converts text into vector representations for retrieval.  
3. **RAG Pipeline** – Retrieves relevant knowledge and generates responses.  
4. **Frontend** – A **Streamlit-based UI** for easy interaction.  

### Architecture:
![Architecture Image](https://github.com/UtkarshSinghShorthillsAI/Rag-ChatBot/blob/main/data_sample/rag_architecture.png)

### **Directory Structure**  
```
mw-bot/
├── data/               # Stores all data used in the pipeline
│   ├── chunks/         # Chunked jsonl files
│   ├── embeddings/     # Vector embeddings
│   ├── processed/      # Cleaned and structured JSON files
│   ├── raw/            # Raw scraped pages from Minecraft Wiki
│   ├── vector_db/      # ChromaDB storage for fast retrieval
│   ├── unique_QnA.json # Ground Truth
│   └── pages.json      # Stores page names collected from the wiki
├── notebooks/          # Jupyter notebooks for testing and debugging
├── requirements.txt   
├── src/                
│   ├── chatbot/    
│   │   └── bot.py      # Streamlit Frontend
│   ├── embedder/       # Beta Version of embedder
│   ├── embedderv2/     # Scalable embedding implementation
│   │   ├── embed.py    # Generate embeddings
│   │   └── vector_store.py  # Store in vector db
│   ├── evaluator/      # Evaluation framework
│   ├── pipeline/
│   │   ├── generator.py # LLM-based response generation
│   │   ├── pipeline.py  # Orchestrates retrieval and generation
│   │   └── retriever.py # Retrieves relevant information from vector DB
│   ├── run_pipeline.py
│   ├── scraper/        # Beta version of Scraper
│   ├── scraperv2/      # Full-scale scraper architecture
│   │   ├── page_collector.py # Collect pages to scrape
│   │   ├── scraper.py       # Scrapes listed pages
│   │   ├── preprocess.py    # Cleans and structure raw scraped data
│   │   └── chunker.py       # Chunks preprocessed data   

```
This bot is **not a real-time scraper**—it **scrapes once, embeds the data, and then answers queries** using the stored knowledge.  
Here’s a **concise yet structured documentation** for the **Scraping** section of your RAG bot:



# **Scraping**  

The scraper extracts structured data from **Minecraft Wiki** to build a knowledge base for retrieval. Instead of scraping the entire wiki, we collect specific **categories of pages**, preprocess their content, and prepare them for embedding.

## **How It Works**  
The scraper runs in **four key stages**:  
1. **Page Collection** → Gather all page names for a given category (e.g., Blocks, Mobs).  
2. **Scraping** → Extract raw data from the wiki pages using Selenium and BeautifulSoup.  
3. **Preprocessing** → Clean, filter, and structure the scraped content.  
4. **Chunking** → Break down text into smaller, meaningful chunks for embeddings.  

---

## **1️⃣ Page Collection** (`page_collector.py`)  
- **Gathers all page names** for a category (e.g., "Blocks") and stores them in `pages.json`.  
- Uses **Selenium** to navigate category pages and extract article links.  
- Supports **pagination handling** to fetch all entries in a category.  

🔹 **Usage Example**  
```python
collector = CategoryPageCollector(category="Blocks")
collector.run()
```
✅ **Output:** `pages.json` (list of page names to scrape)  

---

## **2️⃣ Scraping Wiki Pages** (`scraper.py`)  
- Reads `pages.json` and **scrapes the wiki content** using **Selenium**.  
- Extracts **sections, tables, and crafting recipes** if available.  
- Saves data in **JSON format** inside `data/raw/`.  

🔹 **Usage Example**  
```python
scraper = MinecraftWikiScraper(["Diamond", "Iron_Sword", "Creeper"])
scraper.run()
```
✅ **Output:** `data/raw/<page_name>.json` (structured raw wiki data)  

---

## **3️⃣ Preprocessing** (`preprocess.py`)  
- Cleans text (removes ads, irrelevant content, and unnecessary markup).  
- Filters unwanted sections (e.g., "References", "Gallery").  
- Flattens structured content for easier embedding.  

🔹 **Usage Example**  
```python
preprocessor = Preprocessor()
preprocessor.run()
```
✅ **Output:** `data/processed/<page_name>.json` (cleaned & structured text)  

---

## **4️⃣ Chunking** (`chunker.py`)  
- Splits long text into **overlapping chunks** for embedding.  
- Uses **LangChain's RecursiveCharacterTextSplitter** for intelligent splitting.  
- Generates **unique chunk IDs** to maintain traceability.  

🔹 **Usage Example**  
```python
chunker = Chunker()
chunker.run()
```
✅ **Output:** `data/chunks/<page_name>.jsonl` (chunked text data)  

---

## **Final Output**  
After running all steps, the scraped and processed data is stored in:  
- `data/raw/` → Raw extracted wiki pages.  
- `data/processed/` → Cleaned and structured content.  
- `data/chunks/` → Chunked data ready for embeddings.  

---

This structured approach ensures **high-quality, structured input for retrieval**, making the RAG bot more accurate and efficient. 


# **Embedding**

The **embedding module** is responsible for converting structured text into **dense vector representations** for efficient retrieval. It takes **preprocessed and chunked text**, generates embeddings using a chosen model, and stores them in a **vector database** for the RAG pipeline.

## **How It Works**
The embedding process follows three main steps:
1. **Embedding Generation** → Converts text chunks into vector representations using **bge-base-en embeddings**.
2. **Storage in Vector DB** → Stores embeddings in **ChromaDB** for retrieval.
3. **Efficient Retrieval** → Enables fast, similarity-based searches for relevant information.

---

## **1️⃣ Embedding Generation** (`embed.py`)
- Uses **bge-base-en** (or any compatible model) to generate vector embeddings.
- Reads **chunked text** from `data/chunks/`.
- Saves embeddings in **JSONL format** inside `data/embeddings/`.

🔹 **Usage Example**
```python
embedder = EmbeddingGenerator()
embedder.run()
```
✅ **Output:** `data/embeddings/<page_name>.jsonl` (vectorized text)

---

## **2️⃣ Vector Database Storage** (`vector_store.py`)
- Uses **ChromaDB** to store and retrieve embeddings efficiently.
- Avoids **duplicate storage** by checking existing chunk IDs.
- Supports **metadata storage** for better searchability.

🔹 **Usage Example**
```python
vector_store = VectorStore()
vector_store.run()
```
✅ **Output:** `data/vector_db/` (ChromaDB storage)

---

## **Final Output**
- `data/embeddings/` → JSONL files with text + vector embeddings.
- `data/vector_db/` → ChromaDB storing indexed embeddings for fast retrieval.

This structure allows the bot to **quickly retrieve relevant information**, making the RAG pipeline more efficient. 🚀


---
# **Retrieval-Augmented Generation (RAG) Pipeline Documentation**  

The **RAG pipeline** is the **core** of the Minecraft Wiki Bot, allowing it to answer queries by **retrieving relevant information** and generating responses using an **LLM**. Instead of relying on an LLM’s pre-trained knowledge, the bot **dynamically retrieves information** from the **vector database**, ensuring more **accurate and up-to-date** responses.

---

## **Pipeline Overview**  

The **Minecraft RAG pipeline** consists of three primary components:  

### **1️⃣ Retriever (`retriever.py`)**  
- Queries the **ChromaDB vector database** to fetch the most relevant **wiki chunks** for a given query.  
- Uses **bge-base-en embeddings** to encode queries and perform similarity-based retrieval.  

🔹 **Workflow:**  
1. Generate **query embedding** using Gemini API.  
2. Perform **nearest neighbor search** in the vector database.  
3. Return the **top-k** most relevant chunks and their sources.  

🔹 **Usage Example:**  
```python
retriever = Retriever()
chunks, sources = retriever.query("How do I craft an enchantment table?", top_k=5)
```
✅ **Output:** List of relevant wiki text snippets.  

---

### **2️⃣ Generator (`generator.py`)**  
- Uses **Google Gemini** LLM to generate a **natural language response** based on retrieved chunks.  
- If no relevant chunks are found, the bot responds with `"I don't know."` instead of hallucinating.  
- If a **crafting recipe** is asked and found in the context, it structures the response as a **grid**.  

🔹 **Workflow:**  
1. Format the retrieved **context** into a structured prompt.  
2. Generate a **LLM-based response** strictly using the provided context.  
3. Append a **source link** (if available).  

🔹 **Usage Example:**  
```python
generator = Generator()
response = generator.generate_response("How do I craft an enchantment table?", chunks, sources)
```
✅ **Output:** Fully structured answer with context.  

---

### **3️⃣ RAG Pipeline (`pipeline.py`)**  
- **Orchestrates the full retrieval-generation process.**  
- First, **retrieves relevant context** from the vector database.  
- Then, **generates an answer** using the **retrieved knowledge**.  

🔹 **Workflow:**  
1. Receive a **user query**.  
2. Retrieve **relevant chunks** using the **Retriever**.  
3. Generate a **response** using the **Generator**.  

🔹 **Usage Example:**  
```python
pipeline = RAGPipeline()
response = pipeline.process_query("What is the best fuel source in Minecraft?")
print(response)
```
✅ **Output:** Answer based on **Minecraft Wiki** data.  

---

## **End-to-End RAG Flow**  

1️⃣ **User Query** → `"How do I craft an enchantment table?"`  
2️⃣ **Retriever** → Fetches **relevant wiki content** from vector DB.  
3️⃣ **Generator** → Uses **retrieved context** to create an **LLM-generated answer**.  
4️⃣ **Final Output** →  
```
To craft an enchantment table, you need:
- 4 Obsidian
- 2 Diamonds
- 1 Book
Arrange the materials as follows:
[ O ] [ O ] [ O ]
[ D ] [ O ] [ D ]
[ - ] [ O ] [ - ]

📌 Read more at: [Minecraft Wiki](https://minecraft.wiki)
```

## **Streamlit Frontend**  

The **Streamlit bot** provides a **simple and interactive UI** for users to query the **Minecraft RAG Bot**. Instead of using the pipeline via command line, users can enter questions in a web interface and receive **instant, structured responses**.

---

## **Features**  
✅ **User-Friendly UI** – Enter queries directly in a text box.  
✅ **Real-Time Responses** – Answers fetched from the RAG pipeline.  
✅ **Markdown Formatting** – Structured, easy-to-read answers.  
✅ **Source Links** – Provides a reference to the **Minecraft Wiki**.  

---

## **How It Works**  
1️⃣ **User enters a query** (e.g., `"How do I breed villagers?"`).  
2️⃣ **Query is processed** by the RAG pipeline.  
3️⃣ **Answer is displayed** with structured text and source links.  

---

### **Usage Example**  
### **Run the Streamlit bot**  
```bash
streamlit run src/chatbot/bot.py
```
🔹 **Opens in browser:** `http://localhost:8501`  

### **Example Interaction Screenshot**  
![Minecraft Bot UI](https://github.com/UtkarshSinghShorthillsAI/Rag-ChatBot/blob/main/data_sample/bot_ui.png)




## **Key Code Components**
 **`bot.py`** (Streamlit UI)
- Handles user input.  
- Calls the **RAG pipeline**.  
- Displays the response.  



## **Final Notes**
- The pipeline **only answers based on retrieved knowledge**, reducing **hallucination risks**.  
- If no relevant information is found, it **does not generate misleading answers**.  
- This design ensures **accurate, source-backed responses** for Minecraft-related queries.  


