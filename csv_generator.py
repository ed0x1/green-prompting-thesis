import asyncio, csv, random
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from google.colab import drive
drive.mount('/content/drive')

API_KEY = ""
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
OUTPUT = "/content/drive/MyDrive/risultati_hotpot_500.csv"
SAMPLE_SIZE = 500
CONCURRENCY = 6

client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.together.xyz/v1")
sem = asyncio.Semaphore(CONCURRENCY)


REWRITE_CLEAR = "You are a technical editor. Rewrite the input question to be the shortest, clearest possible version of itself. Output ONLY the rewritten question. Do NOT answer the question. No filler text."
REWRITE_COMPLEX = "You are a verbose academic bureaucrat. Rewrite the input question to use extremely complex vocabulary, archaic phrasing, and convoluted, heavily subordinated sentence structures, while strictly preserving the original meaning. Make it as long-winded and pompous as possible. Output ONLY the rewritten question. Do NOT answer the question. No filler text, no meta-talk."

ANSWER_STD   = "Answer"
ANSWER_SHORT = "Answer with the ABSOLUTE MINIMUM number of words possible."

def clean(t):
    return str(t).strip().replace('\n', ' ').replace('\r', '').replace(';', ',')

async def call(sys, user, max_tok):
    for attempt in range(8):
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.1, max_tokens=max_tok)
            return clean(r.choices[0].message.content), r.usage.completion_tokens
        except Exception as e:
            await asyncio.sleep((2 ** attempt) if "429" in str(e) else 1)
    return "ERROR", 0

async def process(q, gold):
    async with sem:
        q = clean(q)

        clear, complex_ = await asyncio.gather(
            call(REWRITE_CLEAR,   q, 128),
            call(REWRITE_COMPLEX, q, 256),
        )
        clear, complex_ = clear[0], complex_[0]
        if clear == "ERROR" or complex_ == "ERROR": return None
        if not clear.strip() or not complex_.strip(): return None

        res = await asyncio.gather(
            call(ANSWER_STD,   clear,    512),
            call(ANSWER_SHORT, clear,    64),
            call(ANSWER_STD,   complex_, 512),
            call(ANSWER_SHORT, complex_, 64),
        )
        texts  = [r[0] for r in res]
        tokens = [r[1] for r in res]
        return [q, clean(gold), clear, complex_, *texts, *tokens]

async def main():
    print("Caricamento HotpotQA...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)

    data = []
    for row in ds:
        q = row['question']
        gold = row['answer']
        if gold:
            data.append((q, gold))
        if len(data) >= SAMPLE_SIZE:
            break
    print(f"Raccolte {len(data)} domande con gold answer")

    headers = ["Original_Q","Gold_Answer","Clear_Prompt","Complex_Prompt",
               "Response_Clear_Std","Response_Clear_Short","Response_Complex_Std","Response_Complex_Short",
               "Tokens_Clear_Std","Tokens_Clear_Short","Tokens_Complex_Std","Tokens_Complex_Short"]
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f, delimiter=';').writerow(headers)

    print("Elaborazione")
    tasks = [process(q, g) for q, g in data]
    with open(OUTPUT, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        for coro in tqdm.as_completed(tasks, total=len(tasks)):
            r = await coro
            if r: w.writerow(r)

    print(f"\n Salvato in: {OUTPUT}")

await main()