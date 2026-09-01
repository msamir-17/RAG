# eval/run_ragas.py
import json
import sys, os
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings # Added Embeddings
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_correctness
from dotenv import load_dotenv # Added to load API Keys

load_dotenv() # Ensure MISTRAL_API_KEY is loaded

def run_eval(snapshot_path):
    with open(snapshot_path, "r") as f:
        data = json.load(f)
    
    # Deterministic Ground Truth Generation
    # We pick the largest transaction as our "fact check" item
    sorted_tx = sorted(data['report']['transactions'], key=lambda x: x['debit'], reverse=True)
    top_tx = sorted_tx[0]
    
    eval_dict = {
        "question": ["What is the highest single expenditure in the statement?"],
        "answer": [f"The highest expenditure is {top_tx['debit']} for {top_tx['description']}."],
        "contexts": [data['contexts']],
        "ground_truth": [str(top_tx['debit'])]
    }
    
    ds = Dataset.from_dict(eval_dict)
    
    # FIX: Define both LLM and Embeddings for Mistral
    model = ChatMistralAI(model="mistral-small-latest")
    embeddings = MistralAIEmbeddings() 
    
    # Pass 'embeddings' parameter to prevent OpenAI default error
    results = evaluate(
    ds, 
    metrics=[faithfulness, context_recall, answer_correctness], 
    llm=model,
    embeddings=embeddings
)
    
    # Save Baseline
    results_df = results.to_pandas()
    results_df.to_json("eval/baseline_scores.json")
    print("✅ Baseline scores saved to eval/baseline_scores.json")
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        snap_file = sys.argv[1]
    else:
        snap_file = "eval/snapshots/snap_20260901_001928.json"
        
    if os.path.exists(snap_file):
        run_eval(snap_file)
    else:
        print(f"❌ Error: Snapshot file not found at '{snap_file}'")