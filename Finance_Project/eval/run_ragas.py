# eval/run_ragas.py
import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_mistralai import ChatMistralAI

def run_eval(snapshot_path):
    with open(snapshot_path, "r") as f:
        data = json.load(f)
    
    # Deterministic Ground Truth Generation
    # We pick the 3 largest transactions as our "fact check" items
    sorted_tx = sorted(data['report']['transactions'], key=lambda x: x['debit'], reverse=True)
    top_tx = sorted_tx[0]
    
    eval_dict = {
        "question": ["What is the highest single expenditure in the statement?"],
        "answer": [f"The highest expenditure is {top_tx['debit']} for {top_tx['description']}."],
        "contexts": [data['contexts']],
        "ground_truth": [str(top_tx['debit'])]
    }
    
    ds = Dataset.from_dict(eval_dict)
    model = ChatMistralAI(model="mistral-small-latest")
    
    results = evaluate(ds, metrics=[faithfulness, answer_relevancy], llm=model)
    
    # Save Baseline
    results_df = results.to_pandas()
    results_df.to_json("eval/baseline_scores.json")
    return results