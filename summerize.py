import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os
from config import config

sum_prompt = """
請根據提供的醫學資料（報告、論文或書籍）生成結構化摘要，並嚴格按照以下類別進行。摘要應以簡潔的格式:"類別名稱：關鍵信息"列出每個類別下的重要信息。除非與類別直接相關，否則無需額外解釋或詳細描述：

解剖結構：提及任何被特別討論的解剖結構。
身體功能：列出被強調的任何身體功能。
身體測量：包含任何的常規測量，像是血壓或體溫。
測量結果：這些測量的結果。
測量單位：每項測量的單位。
測量值：這些測量的數值。
實驗室數據：概述任何被提及的實驗室試驗。
實驗室結果：這些試驗的結果（例如"增加"、"減少"）。
實驗室數值：試驗的具體數值。
實驗室單位：這些數值的測量單位。
藥物：被討論到的藥物名稱。
藥物劑量、用藥時長、藥物形式、用藥頻率、給藥途徑、用藥狀態、藥物強度、藥物單位、藥物總劑量：為每個藥物屬性提供簡要的資訊。
問題：指出任何醫療狀況或發現。
醫療程序：描述任何醫療程序。
程序結果：這些程序的結果。
程序方法：程序所使用的方法。
嚴重程度：被提到的情況的嚴重程度。
醫療設備：列出任何被使用到的醫療設備。
物質濫用：註明任何被提及的物質濫用情況。

每個類別僅在與醫學資料相關時才需回答。請確保摘要清晰直接，適合快速參考，並使用繁體中文回答。
"""


def split_into_chunks(text, tokens=500):
    # should be changed to gpt-4o?
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
    return chunks


def call_ollama_api(chunk):
    client = OpenAI(
        base_url=f"{config.base_url}/v1",
        api_key="ollama"
    )
    
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": sum_prompt},
            {"role": "user", "content": f" {chunk}"},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].message.content


def process_chunks(content):
    chunks = split_into_chunks(content)

    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_ollama_api, chunks))
    return responses


if __name__ == "__main__":
    content = " sth you wanna test"
    process_chunks(content)

# Can take up to a few minutes to run depending on the size of your data input