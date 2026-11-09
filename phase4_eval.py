#!/usr/bin/env python3
"""
Phase 4b - Run the selected models on original vs perturbed GSM8K questions.

Memorisation test (one-pager point 4): a model that has memorised GSM8K answers does well
on the original test questions but drops when the SAME problem is re-instantiated with new
names/numbers (GSM-Symbolic) or perturbed (GSM-Plus). A model that really reasons keeps
its accuracy. The paired drop is the ground-truth contamination signal.

Items (paired by seed question):
  GSM-Symbolic 'main'  100 GSM8K test questions + K re-instantiations each (default K=4)
  GSM-Plus 'testmini'  300 GSM8K test questions + 'numerical substitution'
                       + 'problem understanding' (paraphrase) versions
Prompt: the standard 8-shot chain-of-thought GSM8K prompt (Wei et al. 2022), greedy
decoding, answer = number after "The answer is" (fallback: last number).

Backends:  --backend vllm  (Linux GPU: Colab / Kaggle; fast)
           --backend hf    (any GPU incl. Windows laptops; slower)
Each model runs in its own subprocess, so an out-of-memory crash only loses that model.
Predictions are appended batch by batch; rerunning the same command resumes.
trust_remote_code is OFF unless --trust-remote-code is given; models that need it are
skipped and logged.

Usage:
  python phase4_eval.py --models out_phase1/phase4/models.csv --tiers local --backend hf
  python phase4_eval.py --models out_phase1/phase4/models.csv --tiers local,colab --backend vllm
  python phase4_eval.py ... --limit 5 --only Qwen/Qwen2-0.5B          (quick test)
"""
import argparse, json, os, re, subprocess, sys, time

FEWSHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.", "6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.", "5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.", "39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.", "8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.", "9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.", "29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.", "33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.", "8"),
]
PREFIX = "".join(f"Question: {q}\nAnswer: {c} The answer is {a}.\n\n" for q, c, a in FEWSHOT)
STOP = ["\n\nQuestion:", "Question:", "\n\n\n"]
NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def prompt(q):
    return f"{PREFIX}Question: {q.strip()}\nAnswer:"


def to_float(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


def extract(text):
    for s in STOP:
        text = text.split(s)[0]
    m = re.findall(r"[Tt]he answer is[:\s]*(" + NUM + ")", text)
    if m:
        return to_float(m[-1])
    nums = re.findall(NUM, text)
    return to_float(nums[-1]) if nums else None


def correct(pred, gold):
    if pred is None or gold is None:
        return False
    if abs(gold - round(gold)) < 1e-9:
        return abs(pred - gold) < 1e-6
    return abs(pred - gold) <= 0.01 + 1e-9          # GSM-Plus decimals such as 4.33


# --------------------------------------------------------------------------- items
def gsm_final(sol):
    m = re.search(r"####\s*(" + NUM + ")", sol)
    return to_float(m.group(1)) if m else None


def load_items(sym_k, limit, cache_dir):
    from huggingface_hub import hf_hub_download
    items = []
    p = hf_hub_download("apple/GSM-Symbolic", "main/test.jsonl", repo_type="dataset", cache_dir=cache_dir)
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    seeds = sorted({r["id"] for r in rows}, key=int)[: limit or None]
    for sid in seeds:
        rs = sorted([r for r in rows if r["id"] == sid], key=lambda r: int(r["instance"]))
        items.append(dict(item_id=f"sym-{sid}-orig", source="symbolic", group=f"sym-{sid}", variant="original",
                          question=rs[0]["original_question"], gold=gsm_final(rs[0]["original_answer"])))
        for r in rs[:sym_k]:
            items.append(dict(item_id=f"sym-{sid}-{r['instance']}", source="symbolic", group=f"sym-{sid}",
                              variant="symbolic", question=r["question"], gold=gsm_final(r["answer"])))
    p = hf_hub_download("qintongli/GSM-Plus", "data/testmini-00000-of-00001.jsonl", repo_type="dataset",
                        cache_dir=cache_dir)
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    seed_ids = {}
    for r in rows:
        seed_ids.setdefault(r["seed_question"], len(seed_ids))
    keep = set(list(seed_ids)[: limit or None])
    done_orig = set()
    for r in rows:
        if r["seed_question"] not in keep:
            continue
        g = seed_ids[r["seed_question"]]
        if g not in done_orig:
            done_orig.add(g)
            items.append(dict(item_id=f"plus-{g}-orig", source="plus", group=f"plus-{g}", variant="original",
                              question=r["seed_question"], gold=to_float(r["seed_answer"])))
        if r["perturbation_type"] in ("numerical substitution", "problem understanding"):
            items.append(dict(item_id=f"plus-{g}-{r['perturbation_type'].split()[0]}", source="plus",
                              group=f"plus-{g}", variant=r["perturbation_type"], question=r["question"],
                              gold=to_float(r["answer"])))
    return [it for it in items if it["gold"] is not None]


# --------------------------------------------------------------------------- generation
def gen_vllm(model_id, prompts, args):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, dtype="float16", max_model_len=args.max_len, gpu_memory_utilization=0.90,
              trust_remote_code=args.trust_remote_code, tensor_parallel_size=args.tp, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new, stop=STOP)
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outs]


def gen_hf_batches(model_id, prompts, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        trust_remote_code=args.trust_remote_code).to(dev).eval()

    class StopAll(StoppingCriteria):
        def __init__(self, n_in): self.n_in = n_in
        def __call__(self, ids, scores, **kw):
            # stop the batch once every sequence has started a new "Question:"
            full = tok.batch_decode(ids[:, self.n_in:], skip_special_tokens=True)
            return all(any(s in t for s in STOP) for t in full)

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b in range(0, len(order), args.batch):
        idx = order[b:b + args.batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len - args.max_new).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 stopping_criteria=StoppingCriteriaList([StopAll(enc.input_ids.shape[1])]))
        texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        yield idx, texts


def safe(model_id):
    return model_id.replace("/", "__")


def run_one(model_id, args):
    items = load_items(args.sym_k, args.limit, args.cache)
    path = os.path.join(args.out, "preds", safe(model_id) + ".jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["item_id"] for l in open(path, encoding="utf-8")}
    todo = [it for it in items if it["item_id"] not in done]
    print(f"[{model_id}] {len(items)} items, {len(todo)} to run", flush=True)
    if not todo:
        return
    prompts = [prompt(it["question"]) for it in todo]
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        def write(i, text):
            it = todo[i]; pred = extract(text)
            f.write(json.dumps({**{k: it[k] for k in ["item_id", "source", "group", "variant", "gold"]},
                                "pred": pred, "correct": correct(pred, it["gold"]),
                                "output": text[:600]}) + "\n")
        if args.backend == "vllm":
            for i, t in enumerate(gen_vllm(model_id, prompts, args)):
                write(i, t)
        else:
            n = 0
            for idx, texts in gen_hf_batches(model_id, prompts, args):
                for i, t in zip(idx, texts):
                    write(i, t)
                f.flush(); n += len(idx)
                print(f"[{model_id}]   {n}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{model_id}] done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="out_phase1/phase4/models.csv")
    ap.add_argument("--tiers", default="local", help="comma list of: local, colab, kaggle")
    ap.add_argument("--only", default=None, help="run a single model id (used internally)")
    ap.add_argument("--out", default="out_phase1/phase4")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--batch", type=int, default=16, help="hf backend batch size")
    ap.add_argument("--tp", type=int, default=1, help="vllm tensor parallel GPUs (Kaggle 2xT4: 2)")
    ap.add_argument("--sym-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="seed questions per dataset (0 = all)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    if args.only:
        run_one(args.only, args)
        return

    import pandas as pd
    m = pd.read_csv(args.models)
    tiers = [t.strip() for t in args.tiers.split(",")]
    todo = m[m.tier.isin(tiers)].model_id.tolist()
    print(f"[eval] {len(todo)} models in tiers {tiers}, backend {args.backend}")
    log = os.path.join(args.out, "eval_status.csv")
    os.makedirs(args.out, exist_ok=True)
    passthru = [a for a in sys.argv[1:]]
    for k, mid in enumerate(todo, 1):
        print(f"\n[eval] ({k}/{len(todo)}) {mid}", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), *passthru, "--only", mid])
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{mid},{'ok' if r.returncode == 0 else 'failed'},{time.strftime('%Y-%m-%d %H:%M')}\n")
        if r.returncode != 0:
            print(f"[eval] {mid} FAILED (exit {r.returncode}) - continuing", flush=True)
    print("[eval] all done")


if __name__ == "__main__":
    main()
