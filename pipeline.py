import asyncio
import json
import logging
import os
import time
from openai import AsyncOpenAI
import backoff
import openai
from dotenv import load_dotenv
load_dotenv()

system_prompt = """You are tasked with solving basic math word problems that have unknown numbers (e.g. number0, number1, etc). You are asked to do it step by step. During each step, you will choose two numbers to perform either addition, subtraction, multiplication, or division. For each step, only respond with the two numbers chosen and the operation to be performed. Then, say if more steps need to be done by saying "Let us continue" or "We are finished"."""
# system_prompt = """You are a math expert, tasked with solving basic math word problems. You are asked to do it step by step. During each step, you will choose two numbers to perform either addition, subtraction, multiplication, or division. Only respond with the two numbers chosen and the operation chosen. If asked if more steps need to be done, respond with "Let us continue". If not, respond with "We are finished"."""
initial_prompt = """Let's do it step by step. First, I will choose these two numbers and operation:"""
next_prompt = """Next, I will choose these two numbers and operation:"""

# we add periods to the end in the training dataset
continue_msg = "Let us continue"
done_msg = "We are finished"

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.getLogger('backoff').addHandler(logging.StreamHandler())


@backoff.on_exception(backoff.expo, openai.RateLimitError)
@backoff.on_exception(backoff.expo, openai.InternalServerError)
async def prompt_openai(messages: list):
    completion = await openai_client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        temperature=1
    )
    response = completion.choices[0].message.content
    return response


async def pipeline_main(question: str, logmsg: str=None):
    msg_history = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": initial_prompt
        }
    ]
    # limit looping for 10 iterations
    for idx in range(0, 10):
        response = await prompt_openai(msg_history)
        try:
            equation, state_msg = response.split("\n")
            msg_history.append({
                "role": "assistant",
                "content": response
            })
            if state_msg.find(continue_msg) != -1:
                msg_history.append({
                    "role": "assistant",
                    "content": next_prompt
                })
            else:
                # remove sentence punctuation
                if equation[-1] == ".":
                    equation = equation[:-1]
                if logmsg: print(logmsg)
                return equation, msg_history
        except Exception as e:
            print(logmsg, "had exception")
            with open("err.txt", "a") as f:
                f.write("ERROR\n")
                f.write(question + "\n")
                f.write(logmsg + "\n")
                f.write(str(e) + "\n")
                f.write("==============\n\n")
                return "-1", msg_history
    print(logmsg, "too many loops")
    return "-1", msg_history


async def test():
    # testing
    equation, msg_history = await pipeline_main("After eating a hearty meal they went to see the Buckingham palace. There, Rachel learned that number0 visitors came to the Buckingham palace that day. If there were number1 visitors the previous day and number2 visitors the day before that how many visited the Buckingham palace within the past number3 days?")
    print(equation)
    with open("output.json", "w") as f:
        json.dump(msg_history, f, indent=4)

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(test())
    print("Total secs:", time.time() - start_time)