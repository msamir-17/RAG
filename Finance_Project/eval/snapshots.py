# eval/snapshot.py
import json
import os
from datetime import datetime

def capture_eval_snapshot(report, vectorstore, metrics, folder="eval/snapshots/"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Grab chunks from memory Chroma
    data = vectorstore.get()
    
    snapshot = {
        "report": report.model_dump(),
        "contexts": data['documents'],
        "metrics": metrics,
        "timestamp": timestamp
    }
    
    path = f"{folder}snap_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return path