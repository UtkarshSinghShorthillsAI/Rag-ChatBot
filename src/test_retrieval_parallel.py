import json
import os
from dotenv import load_dotenv
from multiprocessing import Pool
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.retrieval_eval import RetrievalEvaluator
from src.log_manager import setup_logger
from src.evaluator.evaluation_model import LMStudioEvaluationModel
from src.evaluator.logging import EvaluationLogger

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI")

# Main logger
logger = setup_logger("logs/test_retrieval_parallel.log")
parallel_logger = EvaluationLogger(eval_type="retrieval")

def evaluate_entry(entry):
    try:
        retriever = Retriever()
        generator = Generator()
        evaluation_model = LMStudioEvaluationModel("http://10.99.22.156:1235/v1/chat/completions")
        retrieval_eval = RetrievalEvaluator(retriever, generator, evaluation_model)

        query = entry["question"]
        ground_truth_answer = entry["answer"]

        # Retrieve chunks
        retrieved_chunks, _ = retriever.query(query, top_k=5)

        # Non-LLM metrics
        context_precision = retrieval_eval.compute_context_precision(query, retrieved_chunks)
        context_recall = retrieval_eval.compute_context_recall(query, ground_truth_answer, retrieved_chunks)

        # LLM-based metrics
        try:
            context_precision_llm = retrieval_eval.compute_context_precision_with_llm(query, retrieved_chunks)
            context_recall_llm = retrieval_eval.compute_context_recall_with_llm(query, ground_truth_answer, retrieved_chunks)
            retrieval_precision_llm = retrieval_eval.compute_retrieval_precision_with_llm(query, retrieved_chunks)
            negative_retrieval_llm = retrieval_eval.compute_negative_retrieval_with_llm(query, retrieved_chunks)
        except Exception as e:
            print(f"❌ LLM Evaluation failed for '{query}': {e}")
            context_precision_llm = "FDTKE"
            context_recall_llm = "FDTKE"
            retrieval_precision_llm = "FDTKE"
            negative_retrieval_llm = "FDTKE"

        result = {
            "query": query,
            "ground_truth_answer": ground_truth_answer,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_precision_llm": context_precision_llm,
            "context_recall_llm": context_recall_llm,
            "retrieval_precision_llm": retrieval_precision_llm,
            "negative_retrieval_llm": negative_retrieval_llm
        }

        parallel_logger.log(result)
        return result

    except Exception as e:
        print(f"❌ Fatal error in entry: {entry['question']} → {e}")
        parallel_logger.log_error(entry["question"], str(e))
        return None


if __name__ == "__main__":
    with open("data/ground_truth_qna.json", "r") as f:
        ground_truth_qna = json.load(f)

    print("🚀 Starting parallel retrieval evaluation with 4 workers...")
    with Pool(processes=4) as pool:
        pool.map(evaluate_entry, ground_truth_qna)

    print("✅ All evaluations complete. Saving to Excel...")
    parallel_logger.log_to_excel()
    print("📊 Excel saved. Logs written to logs/test_retrieval_parallel.log")
