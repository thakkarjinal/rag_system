import os
import json
from pathlib import Path
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
DATASET_PATH = Path(__file__).parent / "dataset.json"



from ingestion import ask_llm, retrieve_chunks, build_context, paragraphs, rerank_chunks  # adjust import


with open(DATASET_PATH) as f:
    data = json.load(f)

HF_TOKEN = os.environ["HF_TOKEN"]
llm = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
    model="openai/gpt-oss-120b:groq",
    temperature=0,
    n=1
)

ragas_llm = LangchainLLMWrapper(llm)


questions = []
answers = []
contexts = []
ground_truths = []


for item in data:

    question = item["question"]

    results = retrieve_chunks(paragraphs, question, limit=10)
    reranked = rerank_chunks(question, results, top_k=3)


    context = build_context(reranked)
    answer = ask_llm(context, question)
    ragas_contexts = [item["object"].properties["text"] for item in reranked]
    
    questions.append(question)
    answers.append(answer)
    contexts.append(ragas_contexts)
    ground_truths.append(item["ground_truth"])

print((contexts))


dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})


result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        context_recall
    ],
    llm=ragas_llm
)

print(result)

AVG_FAITHFULNESS_THRESHOLD = 0.9
MIN_FAITHFULNESS_THRESHOLD = 0.8

AVG_RECALL_THRESHOLD = 0.9
MIN_RECALL_THRESHOLD = 0.8

faithfulness_score = result["faithfulness"]
recall_score = result["context_recall"]

avg_faithfulness = sum(faithfulness_score) / len(faithfulness_score)
avg_recall = sum(recall_score) / len(recall_score)

min_faithfulness = min(faithfulness_score)
min_recall = min(recall_score)

print(f"Avg Faithfulness: {avg_faithfulness}")
print(f"Min Faithfulness: {min_faithfulness}")
print(f"Avg Recall: {avg_recall}")
print(f"Min Recall: {min_recall}")


if avg_faithfulness < AVG_FAITHFULNESS_THRESHOLD:
    raise Exception("Average faithfulness below threshold")

if min_faithfulness < MIN_FAITHFULNESS_THRESHOLD:
    raise Exception("Worst-case faithfulness too low")

if avg_recall < AVG_RECALL_THRESHOLD:
    raise Exception("Average recall below threshold")

if min_recall < MIN_RECALL_THRESHOLD:
    raise Exception("Worst-case recall too low")


print("✅ Evaluation passed thresholds")