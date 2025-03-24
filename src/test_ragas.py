import json
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.ragas_eval import RagasEvaluator
from src.log_manager import setup_logger

logger = setup_logger("logs/ragas_eval.log")

# Load Ground Truth QnA
with open("data/ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

retriever = Retriever()
generator = Generator()

results = []

for entry in ground_truth_qna:
    query = entry["question"]
    ground_truth_answer = entry["answer"]

    retrieved_chunks, _ = retriever.query(query, top_k=5)
    generated_answer = generator.generate_response(query,retrieved_chunks, [])

    results.append({
        "query": query,
        "ground_truth_answer": ground_truth_answer,
        "retrieved_chunks": retrieved_chunks,
        "generated_answer": generated_answer
    })

# Run RAGAS evaluation
ragas_evaluator = RagasEvaluator(results)
eval_scores = ragas_evaluator.run()

print("\n\n--- RAGAS METRICS ---")
for k, v in eval_scores.items():
    print(f"{k}: {v:.3f}")

logger.info("RAGAS Evaluation Complete")
logger.info(eval_scores)
