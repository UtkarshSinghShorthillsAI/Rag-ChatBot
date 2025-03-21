import json
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.retrieval_eval import RetrievalEvaluator
from src.evaluator.logging import EvaluationLogger


from src.log_manager import setup_logger

logger = setup_logger("logs/test_retrieval.log")



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

    # ✅ Compute retrieval evaluation metrics (with LLM-based checks)
    context_precision = retrieval_eval.compute_context_precision(query, retrieved_chunks)
    context_recall = retrieval_eval.compute_context_recall(query, ground_truth_answer, retrieved_chunks)
    context_overlap = retrieval_eval.compute_context_overlap(query, ground_truth_answer, retrieved_chunks)
    negative_retrieval = retrieval_eval.compute_negative_retrieval(query, retrieved_chunks)

    # ✅ LLM-based metrics 
    context_precision_llm = retrieval_eval.compute_context_precision_with_llm(query, retrieved_chunks)
    context_recall_llm = retrieval_eval.compute_context_recall_with_llm(query, ground_truth_answer, retrieved_chunks)
    retrieval_precision_llm = retrieval_eval.compute_retrieval_precision_with_llm(query, retrieved_chunks)
    context_overlap_llm = retrieval_eval.compute_context_overlap_with_llm(query, ground_truth_answer, retrieved_chunks)
    negative_retrieval_llm = retrieval_eval.compute_negative_retrieval_with_llm(query, retrieved_chunks)

    # ✅ Log results for this query
    result = {
        "query": query,
        "ground_truth_answer": ground_truth_answer,
        "context_precision":context_precision,
        "context_recall": context_recall,
        "context_overlap": context_overlap,
        "negative_retrieval": negative_retrieval,
        "context_precision_llm": context_precision_llm,
        "context_recall_llm": context_recall_llm,
        "retrieval_precision_llm": retrieval_precision_llm,
        "context_overlap_llm": context_overlap_llm,
        "negative_retrieval_llm": negative_retrieval_llm
    }
    logger.log(result)

# ✅ Convert all logs to Excel (Run separately if needed)
logger.log_to_excel()
