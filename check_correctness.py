import asyncio, csv, os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from google.colab import drive
drive.mount('/content/drive')


API_KEY = ""
JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
INPUT  = "/content/drive/MyDrive/risultati_nq_500.csv"
OUTPUT = "/content/drive/MyDrive/nq_500_giudicato.csv"
CONCURRENCY = 3

client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.together.xyz/v1")
sem = asyncio.Semaphore(CONCURRENCY)

JUDGE_PROMPT = """You are an impartial judge. Decide if the model's response correctly answers the question.

Question: {q}
Reference answer: {gold}
Model response: {resp}

Rules:
- CORRECT if the response conveys the reference answer (aliases, full names, paraphrases, equivalent formulations all count).
- INCORRECT if it gives a different entity, wrong fact, or refuses.
- IGNORE length and style — judge only factual correctness.

Reply with exactly one word: CORRECT or INCORRECT."""

COLS = ["Clear_Std", "Clear_Short", "Complex_Std", "Complex_Short"]


def needs_judge(resp, gold):
    return not (gold.lower() in resp.lower() and len(resp.split()) <= 8)

async def judge(q, gold, resp):
    async with sem:
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=q, gold=gold, resp=resp)}],
                    temperature=0, max_tokens=10)
                out = r.choices[0].message.content.strip().upper()
                return "CORRECT" if "CORRECT" in out and "INCORRECT" not in out else "INCORRECT"
            except Exception:
                await asyncio.sleep(2 ** attempt)
        return "ERROR"


def contingency(verdicts, a, b):
    """Conta le 4 celle confrontando la condizione a con la condizione b,
    SOLO sulle righe in cui entrambi i verdetti sono validi (CORRECT/INCORRECT)."""
    both_ok = only_a = only_b = both_no = skipped = 0
    for v in verdicts:
        va, vb = v.get(a), v.get(b)
        if va not in ("CORRECT", "INCORRECT") or vb not in ("CORRECT", "INCORRECT"):
            skipped += 1
            continue
        oa, ob = (va == "CORRECT"), (vb == "CORRECT")
        if oa and ob:        both_ok += 1
        elif oa and not ob:  only_a  += 1
        elif not oa and ob:  only_b  += 1
        else:                both_no += 1
    return both_ok, only_a, only_b, both_no, skipped

def print_table(title, verdicts, a, b, note=""):
    bo, oa, ob, bn, sk = contingency(verdicts, a, b)
    tot = bo + oa + ob + bn
    accA = bo + oa
    accB = bo + ob
    netto = accB - accA
    pct = lambda x: f"{x/tot*100:4.1f}%" if tot else "  n/a"

    print("\n" + "=" * 64)
    print(title)
    if note:
        print(note)
    print("=" * 64)
    print(f"  {'Caso':<40}{'N':>6}{'%':>8}")
    print(f"  {'-'*54}")
    print(f"  {'Giuste in entrambe (CORRECT->CORRECT)':<40}{bo:>6}{pct(bo):>8}")
    print(f"  {'Solo '+a+' giusta':<40}{oa:>6}{pct(oa):>8}")
    print(f"  {'Solo '+b+' giusta':<40}{ob:>6}{pct(ob):>8}")
    print(f"  {'Sbagliate in entrambe (INCORR->INCORR)':<40}{bn:>6}{pct(bn):>8}")
    print(f"  {'-'*54}")
    print(f"  {'TOTALE righe valide':<40}{tot:>6}")
    if sk:
        print(f"  ({sk} righe escluse per verdetto ERROR/mancante)")
    print(f"  Accuracy {a}: {accA}/{tot}  |  Accuracy {b}: {accB}/{tot}")
    print(f"  Saldo netto (b - a): {netto:+d}  "
          f"({'b peggiora' if netto<0 else 'b migliora' if netto>0 else 'pari'})")
    print(f"  Rotte da a->b: {oa}   |   Riparate da a->b: {ob}")


async def main():
    with open(INPUT, encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    N = len(rows)


    tasks, mapping = [], []
    verdicts = [{} for _ in rows]
    for i, r in enumerate(rows):
        for c in COLS:
            if needs_judge(r[f"Response_{c}"], r["Gold_Answer"]):
                tasks.append(judge(r["Original_Q"], r["Gold_Answer"], r[f"Response_{c}"]))
                mapping.append((i, c))
            else:
                verdicts[i][c] = "CORRECT"
    print(f"Righe: {N} | al giudice: {len(tasks)}\n🚀 Giudicando...")
    for (i, c), v in zip(mapping, await tqdm.gather(*tasks)):
        verdicts[i][c] = v


    with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["Original_Q", "Gold_Answer"] + [f"{c}__verdict" for c in COLS])
        for r, v in zip(rows, verdicts):
            w.writerow([r["Original_Q"], r["Gold_Answer"]] + [v.get(c, "ERROR") for c in COLS])


    print("\n" + "=" * 64)
    print("ACCURACY PER CONDIZIONE")
    print("=" * 64)
    for c in COLS:
        n = sum(1 for v in verdicts if v.get(c) == "CORRECT")
        valid = sum(1 for v in verdicts if v.get(c) in ("CORRECT", "INCORRECT"))
        print(f"  {c:<16} {n}/{valid} ({n/valid*100:.0f}%)" if valid else f"  {c:<16} n/a")


    # Complex_Std -> Clear_Std  (effetto della semplificazione, a parita' di Std)
    print_table(
        "TABELLA 1 — STILE (Complex_Std -> Clear_Std)",
        verdicts, "Complex_Std", "Clear_Std",
        note="Effetto della semplificazione sintattica a parita' di condizione standard.")

    # Complex_Std -> Complex_Short
    print_table(
        "TABELLA 2 — BREVITA' su COMPLEX (Complex_Std -> Complex_Short)",
        verdicts, "Complex_Std", "Complex_Short",
        note="Effetto del vincolo di brevita' sul prompt complesso.")

    # Clear_Std -> Clear_Short
    print_table(
        "TABELLA 3 — BREVITA' su CLEAR (Clear_Std -> Clear_Short)",
        verdicts, "Clear_Std", "Clear_Short",
        note="Effetto del vincolo di brevita' sul prompt pulito.")

    print(f"\n💾 Verdetti salvati in: {OUTPUT}")

await main()
