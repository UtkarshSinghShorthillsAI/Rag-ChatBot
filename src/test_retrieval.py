import json
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.retrieval_eval import RetrievalEvaluator
from src.evaluator.logging import EvaluationLogger

# Initialize retriever, generator, and evaluator
retriever = Retriever()
generator = Generator()
retrieval_eval = RetrievalEvaluator(retriever, generator)
logger = EvaluationLogger(eval_type="retrieval")

# Load Ground Truth QnA
with open("data/ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

# Process all QnA pairs
for idx, qna in enumerate(ground_truth_qna):
    query = qna["question"]
    ground_truth_answer = qna["answer"]

    # ✅ Retrieve Chunks ONCE and pass them to all methods
    retrieved_chunks, _ = retriever.query(query, top_k=5)

    # ✅ Compute retrieval evaluation metrics
    context_precision = retrieval_eval.compute_context_precision(query, retrieved_chunks)
    context_recall = retrieval_eval.compute_context_recall(query, ground_truth_answer, retrieved_chunks)
    context_overlap = retrieval_eval.compute_context_overlap(query, ground_truth_answer, retrieved_chunks)
    negative_retrieval = retrieval_eval.compute_negative_retrieval(query, retrieved_chunks)

    # ✅ Log results for this query
    result = {
        "query": query,
        "ground_truth_answer": ground_truth_answer,
        "context_precision": {"cosine": context_precision},
        "context_recall": {"cosine": context_recall},
        "context_overlap": {"rougeL": context_overlap},
        "negative_retrieval": {"cosine": negative_retrieval}
    }
    logger.log(result)

# ✅ Convert all logs to Excel (Run separately if needed)
logger.log_to_excel()
