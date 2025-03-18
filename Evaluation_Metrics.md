

# **Evaluation Metrics for RAG System**
This document outlines the evaluation metrics used for assessing the performance of the **retrieval system** and the **faithfulness of LLM-generated responses** in our RAG (Retrieval-Augmented Generation) pipeline.

We employ both **non-LLM-based** (cosine similarity, ROUGE-L) and **LLM-based** methods to ensure robust evaluation.

---

## **1️⃣ Retrieval Evaluation Metrics**
Retrieval evaluation ensures that the **retrieved context** is relevant and useful for answering user queries. 

### **🔹 Overview of Retrieval Evaluation Metrics**
| **Metric**                  | **Method Type**  | **Description** | **Scale** | **Higher is Better?** | **Potential Issues Addressed** |
|-----------------------------|-----------------|----------------|-----------|------------------|---------------------------|
| **Context Precision** | Non-LLM (Cosine Similarity) | Measures how relevant the retrieved chunks are to the query. | 0 - 10 | ✅ Yes | Checks **relevance** of retrieved documents. |
| **Context Recall** | Non-LLM (Cosine Similarity) | Measures if retrieved chunks contain all necessary details. | 0 - 10 | ✅ Yes | Ensures **completeness** of retrieval. |
| **Retrieval Precision (LLM)** | LLM-based | LLM judges if the retrieved chunks contain unnecessary information. | 0 - 10 | ✅ Yes | Reduces **irrelevant retrievals**. |
| **Context Overlap Score** | Non-LLM (ROUGE-L) | Measures textual overlap between retrieved chunks and the ground truth. | 0 - 10 | ✅ Yes | Verifies **lexical similarity** with correct answers. |
| **Negative Retrieval Check** | Non-LLM (Cosine Similarity) | Measures the percentage of irrelevant retrieved chunks. | 0 - 10 | ❌ Lower is Better | Detects **unrelated retrievals**. |
| **Context Precision (LLM)** | LLM-based | LLM rates how precisely the retrieved chunks match the query. | 0 - 10 | ✅ Yes | Validates **retrieval quality** with human-like judgment. |
| **Context Recall (LLM)** | LLM-based | LLM determines if retrieved chunks fully cover the necessary information. | 0 - 10 | ✅ Yes | Ensures **retrieved content is complete**. |
| **Context Overlap Score (LLM)** | LLM-based | LLM assesses how much of the ground truth answer is present in retrieved chunks. | 0 - 10 | ✅ Yes | Evaluates **retrieval effectiveness** using LLM insights. |
| **Negative Retrieval Check (LLM)** | LLM-based | LLM determines how many retrieved chunks are irrelevant. | 0 - 10 | ❌ Lower is Better | Identifies **retrieval errors** from a **semantic perspective**. |

---

## **2️⃣ Faithfulness Evaluation Metrics**
Faithfulness evaluation ensures that the **LLM-generated responses** accurately reflect the retrieved documents and do not introduce hallucinations.

### **🔹 Overview of Faithfulness Evaluation Metrics**
| **Metric**                  | **Method Type**  | **Description** | **Scale** | **Higher is Better?** | **Potential Issues Addressed** |
|-----------------------------|-----------------|----------------|-----------|------------------|---------------------------|
| **Answer-Chunk Similarity** | Non-LLM (Cosine Similarity) | Measures semantic similarity between the generated answer and retrieved chunks. | 0 - 10 | ✅ Yes | Ensures the **answer aligns with retrieved context**. |
| **Faithful Coverage** | Non-LLM (ROUGE-L) | Measures how much of the **ground truth answer** is contained in the **generated response**. | 0 - 10 | ✅ Yes | Checks **how well the answer covers the correct information**. |
| **Negative Faithfulness** | Non-LLM (Cosine Similarity) | Measures how much **unverified content** is introduced in the generated response. | 0 - 10 | ❌ Lower is Better | Detects **hallucinations** and fabrication. |
| **Faithfulness Score (LLM)** | LLM-based | LLM evaluates how **faithful** the generated response is to retrieved chunks. | 0 - 10 | ✅ Yes | Human-like judgment of **faithfulness**. |
| **LLM-Based Faithful Coverage** | LLM-based | LLM evaluates how well the generated response **matches the ground truth answer**. | 0 - 10 | ✅ Yes | Ensures the **answer is grounded** in correct information. |

---

## **3️⃣ How the Evaluation Works**
1. **Retrieval Evaluation:**
   - **Step 1:** Retrieve relevant chunks for the query.
   - **Step 2:** Evaluate using **cosine similarity, ROUGE-L, and LLM-based judgments**.
   - **Step 3:** Identify irrelevant retrievals and check **overlap with ground truth**.

2. **Faithfulness Evaluation:**
   - **Step 1:** Retrieve chunks **once per query**.
   - **Step 2:** Generate LLM response **once per query**.
   - **Step 3:** Evaluate **semantic alignment, overlap, hallucinations, and faithfulness**.
   - **Step 4:** Use **LLM to verify faithfulness** from a **human perspective**.

---

## **4️⃣ Logging and Scaling**
- All evaluation metrics are scaled **between 0 and 10**.
- A higher score is **better** for **precision, recall, overlap, and faithfulness**.
- A lower score is **better** for **negative retrieval and negative faithfulness**.
- Logs are stored in **JSON & CSV formats** for **performance tracking and debugging**.

---

## **5️⃣ Why Use Both Non-LLM and LLM-Based Methods?**
| **Approach** | **Strengths** | **Weaknesses** |
|-------------|--------------|----------------|
| **Non-LLM Based (Cosine, ROUGE-L)** | Fast, scalable, and interpretable. Can be used in batch processing. | May not fully capture semantic meaning, especially with paraphrased content. |
| **LLM-Based Evaluation** | Provides human-like judgment on precision, recall, and faithfulness. Handles nuanced interpretations. | Slower due to API calls and requires fine-tuning for reliable results. |

---

## **6️⃣ Running Evaluation Tests**
### **🔹 Running Retrieval Evaluation**
```bash
python3 -m src.test_retrieval
```
- Evaluates all **retrieval performance metrics** for queries in `ground_truth_qna.json`.
- Results are logged automatically.

### **🔹 Running Faithfulness Evaluation**
```bash
python3 -m src.test_faithfulness
```
- Evaluates all **faithfulness metrics** for generated responses.
- Uses **cosine similarity, ROUGE-L, and LLM-based assessments**.

---

## **7️⃣ Example Output**
```plaintext
🔍 Running Retrieval Evaluation for Query: How do I tame a wolf?
📊 Context Precision: 8.90
📊 Context Recall: 7.50
📊 Retrieval Precision (LLM): 9.20
📊 Context Overlap Score (ROUGE-L): 8.40
📊 Negative Retrieval Score: 1.20
🤖 Context Precision (LLM): 9.00
🤖 Context Recall (LLM): 8.50
🤖 Negative Retrieval Score (LLM): 2.10

🔍 Running Faithfulness Evaluation for Query: How do I tame a wolf?
📊 Answer-Chunk Similarity: 8.90
📊 Faithful Coverage (ROUGE-L): 7.50
📊 Negative Faithfulness: 2.30
🤖 Faithfulness Score (LLM): 9.00
🤖 LLM-Based Faithful Coverage: 8.80

✅ Evaluation Complete! Results logged.
```

---

## **8️⃣ Summary**
| **Component** | **Evaluated Properties** | **Key Metrics Used** |
|--------------|----------------------|------------------|
| **Retriever Evaluation** | Relevance, completeness, retrieval accuracy | Context Precision, Context Recall, Retrieval Precision, Context Overlap, Negative Retrieval |
| **Faithfulness Evaluation** | Answer correctness, grounding, hallucination detection | Answer-Chunk Similarity, Faithful Coverage, Negative Faithfulness, LLM-Based Judgments |

---

### **What's Next?**
- **Fine-tune LLM prompts** for better judgment.
- **Analyze results** and adjust retrieval models **if needed**.
- **Extend evaluation to new datasets** for improved benchmarking.
