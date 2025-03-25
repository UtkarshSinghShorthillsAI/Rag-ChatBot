import json
import os
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.ragas_eval import RagasEvaluator
from src.log_manager import setup_logger
logger = setup_logger("logs/ragas_eval.log")
from dotenv import load_dotenv

load_dotenv()

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
    generated_answer, _ = generator.generate_response(query, retrieved_chunks, [])

    results.append({
        "query": query,
        "ground_truth_answer": ground_truth_answer,
        "retrieved_chunks": retrieved_chunks,
        "generated_answer": generated_answer
    })

# Initialize evaluator with OpenAI
ragas_evaluator = RagasEvaluator(
    results,
    use_openai=False,
    openai_key=os.getenv("OPENAI_API_KEY")
)

# Run evaluation
eval_scores = ragas_evaluator.run()

# Print results
print("\n\n--- RAGAS METRICS ---")
for i, score_dict in enumerate(eval_scores.scores):
    print(f"\nQA Pair {i+1}:")
    for metric, value in score_dict.items():
        print(f"{metric}: {value:.3f}")

logger.info("RAGAS Evaluation Complete")
logger.info(eval_scores)

# Generate Excel report
ragas_evaluator.json_to_excel()