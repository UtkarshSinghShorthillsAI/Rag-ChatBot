import json
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.faithfulness_eval import FaithfulnessEvaluator

# ✅ Initialize retriever, generator, and evaluator
retriever = Retriever()
generator = Generator()
faithfulness_eval = FaithfulnessEvaluator(retriever, generator)

# ✅ Load Ground Truth QnA
with open("data/ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

# ✅ Process each query
for i, qna in enumerate(ground_truth_qna):
    query = qna["question"]
    ground_truth_answer = qna["answer"]

    print(f"\n🔍 Running Faithfulness Evaluation for Query {i+1}/{len(ground_truth_qna)}: {query}")

    # ✅ Evaluate faithfulness in a single function call
    result = faithfulness_eval.evaluate_faithfulness(query, ground_truth_answer, top_k=5)

    # ✅ Log results if evaluation was successful
    if result:
        faithfulness_eval.logger.log(result)

print("\n✅ Faithfulness Evaluation Complete! Results are logged.")
