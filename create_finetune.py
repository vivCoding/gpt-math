import os
import json
import random
import re
import csv
from pipeline import system_prompt, initial_prompt, done_msg, continue_msg, next_prompt

def main():
    # NOTE change to train or dev
    phase = "train"

    with open(f"data/mawps-steps-{phase}.json", "r") as f:
        data = json.load(f)
    
    finetune_entries = []
    for entry in data:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": entry["question"],
            },
            {
                "role": "assistant",
                "content": initial_prompt,
            },
        ]
        num_steps = len(entry["steps"])
        for idx, step in enumerate(entry["steps"]):
            message = step + "\n"
            if idx == num_steps - 1:
                message += done_msg
                messages.append({
                    "role": "assistant",
                    "content": message
                })
                break
            message += continue_msg
            messages.append({
                "role": "assistant",
                "content": message
            })
            messages.append({
                "role": "assistant",
                "content": next_prompt,
            })
        finetune_entries.append({"messages": messages})

    random.shuffle(finetune_entries)
    with open(f"data/ft-mawps-{phase}.jsonl", "w") as f:
        for entry in finetune_entries:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()