import os
import json
import random
import re
import csv

operations = {"+", "-", "*", "/"}


def create_mawps_dataset():
    data = []
    for idx in range(0, 5):
        with open(f"data/orig/cv_mawps/fold{idx}/dev.csv", "r") as f:
            data += list(csv.reader(f))[1:]

    new_data = []
    for idx, entry in enumerate(data):
        question, numbers, equation, answer, group_nums, body, q_stmt = entry
        entry_id = f"mawps-{idx}"
        for idx, number in enumerate(numbers.split(" ")):
            body = body.replace(f"number{idx}", number).replace(" .", ".").replace(" ,", ",").replace(" ?", "?").strip()
            q_stmt = q_stmt.replace(f"number{idx}", number).replace(" .", ".").replace(" ,", ",").replace(" ?", "?").strip()
            equation = equation.replace(f"number{idx}", number)
        
        new_data.append({
            "ID": entry_id,
            "Body": body,
            "Question": q_stmt,
            "Equation": equation,
            "Answer": answer,
        })
        
    with open("data/mawps.json", "w") as f:
        json.dump(new_data, f, indent=4)

    random.shuffle(new_data)
    with open("data/mawps_sampled.json", "w") as f:
        json.dump(new_data[:100], f, indent=4)


def create_svamp_dataset():
    with open("data/orig/SVAMP.json", "r") as f:
        data = json.load(f)

    new_data = []
    for idx, entry in enumerate(data):
        # just renaming id soomthing readable
        new_question = entry["Body"].strip()
        if new_question[-1] != "." and new_question[-1] != "?":
            entry["Question"] = entry["Question"][0].lower() + entry["Question"][1:]
        new_question += " " + entry["Question"]

        ct = 0
        for word in entry["Equation"].split():
            if word in operations:
                ct += 1

        new_entry = {
            "task_id": f"svamp-{idx}",
            "question": new_question,
            "equation": entry["Equation"],
            "answer": entry["Answer"],
            "type": entry["Type"],
            "step_ct": ct
        }
        new_data.append(new_entry)

    with open("data/svamp.json", "w") as f:
        json.dump(new_data, f, indent=4)
    
    random.shuffle(new_data)
    with open("data/svamp_sampled.json", "w") as f:
        json.dump(new_data[:100], f, indent=4)



def parse_eq(eq: str, numbers: dict):
    words = eq.split(" ")
    stack = []
    steps = []
    nums = {k: v for k,v in numbers.items()}
    for word in words:
        if word not in nums and word not in operations:
            nums[word] = float(word)
        stack.append(word)
        if len(stack) > 2 and stack[-1] not in operations and stack[-2] not in operations and stack[-3] in operations:
            t1 = stack.pop()
            t2 = stack.pop()
            op = stack.pop()
            if t2 < t1:
                t1, t2, = t2, t1
            if (op == "-" or op == "/") and (float(nums[t2]) > float(nums[t1])):
                t1, t2 = t2, t1
            term = f"( {t1} {op} {t2} )"
            stack.append(term)
            steps.append(term[1:-1].strip())
            # update numbers dict
            if op == "+": nums[term] = float(nums[t1]) + float(nums[t2])
            if op == "-": nums[term] = float(nums[t1]) - float(nums[t2])
            if op == "*": nums[term] = float(nums[t1]) * float(nums[t2])
            if op == "/": nums[term] = float(nums[t1]) / float(nums[t2])
    if len(stack) > 2 and stack[-1] not in operations and stack[-2] not in operations and stack[-3] in operations:
        t1 = stack.pop()
        t2 = stack.pop()
        op = stack.pop()
        if (op == "-" or op == "/") and (float(nums[t2]) > float(nums[t1])):
            t1, t2 = t2, t1
        term = f"( {t1} {op} {t2} )"
        stack.append(term)
        steps.append(term[1:-1].strip())
    final_eq = stack[0][1:-1].strip()
    return final_eq, steps


def create_mawps_steps_datasets():
    with open("data/orig/cv_mawps/fold0/train.csv", "r") as f:
        data = list(csv.reader(f))[1:]

    new_data = []
    for idx, entry in enumerate(data):
        question, numbers, equation, answer, group_nums, body, q_stmt = entry
        new_data.append({
            "task_id": f"mawps-{idx}",
            "question": question.replace(" .", ".").replace(" ,", ",").replace(" ?", "?"),
            "numbers": {f"number{idx}": num for idx, num in enumerate(numbers.split(" "))},
            "equation": equation,
            "answer": answer
        })

    for idx, entry in enumerate(new_data):
        eq, steps = parse_eq(entry["equation"], entry["numbers"])
        entry["equation"] = eq
        entry["steps"] = steps

    with open("data/mawps-steps-train.json", "w") as f:
        json.dump(new_data, f, indent=4)

    
    with open("data/orig/cv_mawps/fold0/dev.csv", "r") as f:
        data = list(csv.reader(f))[1:]

    new_data = []
    for idx, entry in enumerate(data):
        question, numbers, equation, answer, group_nums, body, q_stmt = entry
        new_data.append({
            "task_id": f"mawps-{idx}",
            "question": question.replace(" .", ".").replace(" ,", ",").replace(" ?", "?"),
            "numbers": {f"number{idx}": num for idx, num in enumerate(numbers.split(" "))},
            "equation": equation,
            "answer": answer
        })

    for idx, entry in enumerate(new_data):
        eq, steps = parse_eq(entry["equation"], entry["numbers"])
        entry["equation"] = eq
        entry["steps"] = steps

    with open("data/mawps-steps-dev.json", "w") as f:
        json.dump(new_data, f, indent=4)

def create_mawps_dev():
    with open("./data/mawps-steps-dev.json", "r") as f:
        data = json.load(f)

    for idx, entry in enumerate(data):
        for num_str, num_val in entry["numbers"].items():
            entry["question"] = entry["question"].replace(num_str, num_val)

    with open("data/mawps-dev.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # create_mawps_dataset()
    # create_svamp_dataset()
    # create_mawps_steps_datasets()
    create_mawps_dev()