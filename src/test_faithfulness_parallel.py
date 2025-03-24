import json
import os
from dotenv import load_dotenv
from multiprocessing import Pool
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.faithfulness_eval import FaithfulnessEvaluator
from src.log_manager import setup_logger
from src.evaluator.evaluation_model import LMStudioEvaluationModel
from src.evaluator.logging import EvaluationLogger

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI")

# Main logger
logger = setup_logger("logs/test_faithfulness_parallel.log")
parallel_logger = EvaluationLogger(eval_type="faithfulness")

def evaluate_entry(entry):
    try:
        retriever = Retriever()
        generator = Generator()
        evaluation_model = LMStudioEvaluationModel(api_url="http://10.99.22.156:1235/v1/chat/completions")
        faithfulness_eval = FaithfulnessEvaluator(retriever, generator, evaluation_model)

        query = entry["question"]
        ground_truth_answer = entry["answer"]
        retrieved_chunks, _ = retriever.query(query, top_k=5)
        generated_answer, _ = generator.generate_response(query, retrieved_chunks, [])

        # Non-LLM metrics
        blobwise = faithfulness_eval.compute_blobwise_similarity(query, retrieved_chunks, generated_answer)
        chunkwise = faithfulness_eval.compute_chunkwise_similarity(generated_answer, retrieved_chunks)
        coverage = faithfulness_eval.compute_faithful_coverage(ground_truth_answer, generated_answer)

        # LLM-based metrics
        try:
            judge_score = faithfulness_eval.llm_as_judge(query, retrieved_chunks, generated_answer)
            coverage_llm = faithfulness_eval.llm_faithful_coverage(query, ground_truth_answer, generated_answer)
        except Exception as e:
            judge_score = "FDTKE"
            coverage_llm = "FDTKE"
            print(f"❌ LLM Evaluation failed for '{query}': {e}")

        result_data = {
            "query": query,
            "ground_truth_answer": ground_truth_answer,
            "generated_answer": generated_answer,
            "blobwise_answer_similarity": blobwise,
            "avg_chunkwise_answer_similarity": chunkwise['avg_chunkwise_score'],
            "max_chunkwise_answer_similarity": chunkwise['max_chunkwise_score'],
            "faithful_coverage": coverage,
            "faithfulness_score_llm": judge_score,
            "faithful_coverage_llm": coverage_llm
        }

        parallel_logger.log(result_data)
        return result_data

    except Exception as e:
        print(f"❌ Fatal error in entry: {entry['question']} → {e}")
        parallel_logger.log_error(entry["question"], str(e))
        return None


if __name__ == "__main__":
    with open("data/ground_truth_qna.json", "r") as f:
        ground_truth_qna = json.load(f)

    print("🚀 Starting parallel faithfulness evaluation with 4 workers...")
    with Pool(processes=4) as pool:
        pool.map(evaluate_entry, ground_truth_qna)

    print("✅ All evaluations complete. Saving to Excel...")
    parallel_logger.log_to_excel()
    print("📊 Excel saved. Logs written to logs/test_faithfulness_parallel.log")
