import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness
)

from app.rag_chatbot import RAGChatbot


def build_dataset(bot, questions):
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for q in questions:
        response, docs = bot.ask(q)

        contexts = [d.page_content for d in docs]

        data["question"].append(q)
        data["answer"].append(str(response))
        data["contexts"].append(contexts)

        # NOTE: Ideally use curated ground truth
        data["ground_truth"].append(str(response))  # placeholder

    return Dataset.from_dict(data)


def run_ragas_evaluation():
    bot = RAGChatbot()

    questions = [
        "Explain embeddings",
        "What tools are used?",
        "Summarize the document"
    ]

    dataset = build_dataset(bot, questions)

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness
        ]
    )

    print("\nRAGAS Evaluation Results:")
    print(result)

    return result


if __name__ == "__main__":
    run_ragas_evaluation()