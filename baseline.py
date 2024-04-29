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

system_prompt = "You are tasked with solving basic math word problems. Respond with only the answer."

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.getLogger('backoff').addHandler(logging.StreamHandler())


@backoff.on_exception(backoff.expo, openai.RateLimitError)
@backoff.on_exception(backoff.expo, openai.InternalServerError)
async def prompt_openai(messages: list):
    completion = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        # model="gpt-4-turbo",
        messages=messages,
        temperature=1
    )
    response = completion.choices[0].message.content
    return response

async def baseline_main(question: str, logmsg: str=None):
    msg_history = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        },
    ]
    response = await prompt_openai(msg_history)
    # remove sentence punctuation
    if response[-1] == ".":
        response = response[:-1]
    if logmsg: print(logmsg)
    return response, []
