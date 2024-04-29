import asyncio
import os
import json
from pathlib import Path
import re
import time
from typing import Dict, List
from dotenv import load_dotenv
from baseline import baseline_main

from pipeline import pipeline_main
load_dotenv()

def load_dataset(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def evaluate_math(equation: str, numbers: Dict[str, float]):
    math_stuff = {"+", "-", "*", "/", "(", ")"}
    equation = equation.replace("$", "")
    clean_equation = []
    for word in equation.split(" "):
        if word in math_stuff:
            clean_equation.append(word)
        else:
            try:
                x = float(word)
                clean_equation.append(str(x))
            except:
                if word in numbers:
                    clean_equation.append(str(numbers[word]))
    
    return float(eval(" ".join(clean_equation)))


async def experiment(dataset: List[Dict], results_dir: str, experiment_name: str, use_pipeline=True, load_from_dir=False):
    # create results dir if not exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    msg_history_dir = os.path.join(results_dir, "msg_histories")
    Path(msg_history_dir).mkdir(parents=True, exist_ok=True)
    eval_logs_dir = os.path.join(results_dir, "eval_logs")
    Path(eval_logs_dir).mkdir(parents=True, exist_ok=True)

    equations = []
    batch_size = 10
    if load_from_dir:
        # if already got gpt responses, just load them and evaluate
        for entry in dataset:
            with open(os.path.join(msg_history_dir, f"{entry['task_id']}.json"), "r") as f:
                data = json.load(f)
                equations.append(data["equation"])
    else:
        for idx in range(0, len(dataset), batch_size):
            if use_pipeline:
                api_calls = [pipeline_main(entry["question"], logmsg=f"got {jdx} {entry['task_id']}") for jdx, entry in enumerate(dataset[idx:idx + batch_size])]
            else:
                api_calls = [baseline_main(entry["question"], logmsg=f"got {jdx} {entry['task_id']}") for jdx, entry in enumerate(dataset[idx:idx + batch_size])]
            responses = await asyncio.gather(*api_calls)
            for jdx, (equation, msg_history) in enumerate(responses):
                entry = dataset[idx + jdx]
                with open(os.path.join(msg_history_dir, f"{entry['task_id']}.json"), "w") as f:
                    json.dump({
                        "equation": equation,
                        "msg_history": msg_history
                    }, f, indent=4)
                equations.append(equation)
            print(f"Got {idx + batch_size} / {len(dataset)}")

    exceptions = []
    success = []
    failed = []
    step_total: Dict[int, int] = {}
    step_success: Dict[int, int] = {}
    for idx, entry in enumerate(dataset):
        # setup step count stuff
        if entry.get("step_ct", None) is not None:
            step_ct = entry["step_ct"]
        else:
            step_ct = len(entry["steps"])
        if step_total.get(step_ct, None) is None:
            step_total[step_ct] = 0
        if step_success.get(step_ct, None) is None:
            step_success[step_ct] = 0
        step_total[step_ct] += 1
        # actually evaluate
        # log it down
        equation = equations[idx]
        with open(os.path.join(eval_logs_dir, entry["task_id"] + ".log"), "w") as f:
            try:
                pred = evaluate_math(equation, entry.get("numbers", {}))
                if pred == entry["answer"]:
                    success.append(entry["task_id"])
                    step_success[step_ct] += 1
                else:
                    failed.append(entry["task_id"])
                f.write(f"{pred == entry['answer']}\n")
                f.write(f"Actual: {pred}\n")
                f.write(f"Expected: {entry['answer']}\n")
            except Exception as e:
                exceptions.append(entry["task_id"])
                f.write(str(e))

    print("=========")
    print(f"Summary ({experiment_name})")
    print("-----")
    print("Success:", len(success))
    print("Failed:", len(failed))
    print("Exceptioned:", len(exceptions))
    print("Total:", len(dataset))
    print("-----")
    print("Accuracy:", len(success) / len(dataset))

    step_stats = {}
    for step_ct, step_success_ct in step_success.items():
        step_stats[step_ct] = {
            "success": step_success_ct,
            "total": step_total[step_ct],
            "accuracy": step_success_ct / step_total[step_ct]
        }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump({
            "stats": {
                "success": len(success),
                "failed": len(failed),
                "exceptioned": len(exceptions),
                "total": len(dataset),
                "accuracy": len(success) / len(dataset),
            },
            "stats_per_step_ct": step_stats,
            "tasks": {
                "succeeded": success,
                "failed": failed,
                "exceptioned": exceptions
            }
        }, f, indent=4)


async def main():
    # TODO change this to approporate settings
    dataset_name = "mawps-dev"
    # dataset_name = "mawps-steps-dev"
    # dataset_name = "svamp-masked"
    # dataset_name = "svamp"
    use_pipeline = True
    # if already got gpt responses, just load them and evaluate
    load_from_dir = True

    print("Loading dataset...")
    dataset = load_dataset(f"./data/{dataset_name}.json")
    await experiment(
        dataset=dataset,
        results_dir=f"./results/{'pipeline' if use_pipeline else 'baseline'}/{dataset_name}/",
        experiment_name=dataset_name + f"({'pipeline' if use_pipeline else 'baseline'})",
        use_pipeline=use_pipeline,
        load_from_dir=load_from_dir
    )

if __name__ == "__main__":
    start_time = time.time()
    print("Starting...")
    asyncio.run(main())
    print()
    print ("Total secs:", time.time() - start_time)