import json
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.faithfulness_eval import FaithfulnessEvaluator
from src.evaluator.logging import EvaluationLogger

# Initialize retriever, generator, and evaluator
retriever = Retriever()
generator = Generator()
faithfulness_eval = FaithfulnessEvaluator(retriever, generator)
logger = EvaluationLogger(eval_type="faithfulness")

# Load Ground Truth QnA
with open("data/ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

# Process all QnA pairs
for idx, qna in enumerate(ground_truth_qna):
    query = qna["question"]
    ground_truth_answer = qna["answer"]

    # ✅ Retrieve Chunks ONCE
    retrieved_chunks, _ = retriever.query(query, top_k=5)
    generated_answer = generator.generate_response(query, retrieved_chunks, [])

    # ✅ Compute faithfulness evaluation metrics
    answer_similarity = faithfulness_eval.answer_chunk_similarity(query, retrieved_chunks, generated_answer)
    faithful_coverage = faithfulness_eval.compute_faithful_coverage(query, ground_truth_answer, generated_answer)
    negative_faithfulness = faithfulness_eval.compute_negative_faithfulness(query, retrieved_chunks, generated_answer)

    # ✅ Log results for this query
    result = {
        "query": query,
        "ground_truth_answer": ground_truth_answer,
        "generated_answer": generated_answer,
        "answer_chunk_similarity": answer_similarity,
        "faithful_coverage": faithful_coverage,
        "negative_faithfulness": negative_faithfulness
    }
    logger.log(result)

# ✅ Convert all logs to Excel (Run separately if needed)
logger.log_to_excel()
