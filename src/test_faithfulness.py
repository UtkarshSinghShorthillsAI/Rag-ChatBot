import json
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.faithfulness_eval import FaithfulnessEvaluator
import logging
from src.log_manager import setup_logger
from src.evaluator.evaluation_model import ChatGPTEvaluationModel, LMStudioEvaluationModel
from dotenv import load_dotenv
import os
from multiprocessing import Pool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI")


logger = setup_logger("logs/test_faithfulness.log")


# Initialize retriever, generator, and evaluator
retriever = Retriever()
generator = Generator()
# evaluation_model = ChatGPTEvaluationModel(api_key=OPENAI_API_KEY) 
evaluation_model = LMStudioEvaluationModel()
faithfulness_eval = FaithfulnessEvaluator(retriever, generator, evaluation_model)

# Load Ground Truth QnA
with open("data/ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

# Run evaluation for each query in the ground truth QnA
for entry in ground_truth_qna:
    query = entry["question"]
    ground_truth_answer = entry["answer"]

    # Retrieve Chunks ONCE and pass them to all methods
    retrieved_chunks, _ = retriever.query(query, top_k=5)

    # Generate the answer for the given query
    generated_answer, _ = generator.generate_response(query, retrieved_chunks, [])

    # Compute faithfulness evaluation metrics (Non-LLM)
    blobwise_answer_similarity = faithfulness_eval.compute_blobwise_similarity(query, retrieved_chunks, generated_answer)
    chunkwise_answer_similarity = faithfulness_eval.compute_chunkwise_similarity(generated_answer, retrieved_chunks)
    faithful_coverage = faithfulness_eval.compute_faithful_coverage(ground_truth_answer, generated_answer)
    # negative_faithfulness = faithfulness_eval.compute_negative_faithfulness(query, retrieved_chunks, generated_answer)

    # Compute LLM-based faithfulness evaluation metrics, with error handling for API exhaustion
    try:
        faithfulness_score_llm = faithfulness_eval.llm_as_judge(query, retrieved_chunks, generated_answer)
        faithful_coverage_llm = faithfulness_eval.llm_faithful_coverage(query, ground_truth_answer, generated_answer)
    except Exception as e:
        logger.error(f"❌ Error generating response for LLM-based evaluation: {e}")
        faithfulness_score_llm = "FDTKE"  # Failed due to key exhaustion
        faithful_coverage_llm = "FDTKE"  # Failed due to key exhaustion

    # Prepare the result data for logging
    result_data = {
        "query": query,
        "ground_truth_answer": ground_truth_answer,
        "generated_answer": faithfulness_eval.generator.generate_response(query, retrieved_chunks, []),
        "blobwise_answer_similarity": blobwise_answer_similarity,
        "avg_chunkwise_answer_similarity": chunkwise_answer_similarity['avg_chunkwise_score'],
        "max_chunkwise_answer_similarity": chunkwise_answer_similarity['max_chunkwise_score'],
        "faithful_coverage": faithful_coverage,
        # "negative_faithfulness": negative_faithfulness,
        "faithfulness_score_llm": faithfulness_score_llm,
        "faithful_coverage_llm": faithful_coverage_llm
    }

    # Log the results
    faithfulness_eval.logger.log(result_data)

logger.info("Faithfulness Evaluation completed!")
faithfulness_eval.logger.log_to_excel()
logger.info("Results saved to excel.")
