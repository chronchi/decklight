"""Deck-builder semantic search: query-encoder quality per ONNX quantisation (docs/design/deck-builder.md §Search).

Builds (once, LLM-cached) a query set: for 60 cards common in decks/pool, DeepSeek writes a player-style description
without the card's name ("one") and a two-property one ("two"). Each query is encoded with Qwen3-Embedding-0.6B ONNX
variants (fp16 = reference, int8, q4f16), with and without the query instruction, and ranked against the card
embeddings the app ships (data/cards/emb/qwen3-0.6b-d512.npz). Also BM25 over card text as the baseline to beat.

    uv run --with onnxruntime==1.31.1 --with tokenizers==0.22.1 python -u scripts/search/semantic_eval.py \
        --onnx-dir <dir with tokenizer.json and onnx/model_*.onnx> [--n 60]
"""

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from decklight.cards.embed import card_texts
from decklight.demos.llm import complete

INSTRUCT = "Instruct: Given a player's description of a Yu-Gi-Oh! card, retrieve the card whose text matches\nQuery:"
LLM = "opencode-go/deepseek-v4.1-flash"
PROMPT = """For each Yu-Gi-Oh! card below, write two search queries a player might type into a deck builder when they
want a card like it but don't remember its name. Never use the card's name, archetype name or quoted card names.
Use plain player words, 4-14 words, not the card's exact wording.
- "one": describes its single most distinctive effect.
- "two": asks for two of its properties joined by "and" (e.g. "a light tuner that special summons itself and searches").
Answer with JSON only: {{"<id>": {{"one": "...", "two": "..."}}, ...}}

{cards}"""


def fail(msg: str) -> None:
    print(f"ALERT semantic_eval: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def pool_cards(texts: dict[int, tuple[str, str]], n: int) -> list[int]:
    freq: collections.Counter[int] = collections.Counter()
    for ydk in sorted((ROOT / "decks/pool").glob("*.ydk")):
        seen = {int(x) for x in re.findall(r"^(\d+)\s*$", ydk.read_text(errors="ignore"), re.MULTILINE)}
        freq.update(c for c in seen if c in texts and len(texts[c][1]) > 40)
    by_name: dict[str, int] = {}
    for c, _ in freq.most_common():
        by_name.setdefault(texts[c][0], c)
    top = list(by_name.values())[:300]
    return sorted(random.Random(0).sample(top, n))


def make_queries(codes: list[int], texts: dict[int, tuple[str, str]], out: Path) -> list[dict]:
    if out.exists():
        return json.loads(out.read_text())
    rows = []
    for i in range(0, len(codes), 10):
        chunk = codes[i : i + 10]
        cards = "\n\n".join(f"id {c}: {texts[c][0]}\n{texts[c][1]}" for c in chunk)
        raw = complete(PROMPT.format(cards=cards), model=LLM)
        got = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        for c in chunk:
            q = got.get(str(c))
            if not q or not q.get("one") or not q.get("two"):
                fail(f"LLM gave no queries for {c}")
            rows += [{"code": c, "kind": k, "q": q[k]} for k in ("one", "two")]
        print(f"queries {min(i + 10, len(codes))}/{len(codes)}", flush=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    return rows


class Encoder:
    def __init__(self, onnx_dir: Path, variant: str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(onnx_dir / f"onnx/model_{variant}.onnx"), so)
        self.past = [i for i in self.sess.get_inputs() if i.name.startswith("past_key_values")]
        self.ftype = np.float16 if "float16" in self.past[0].type else np.float32

    def __call__(self, text: str, dim: int = 512) -> np.ndarray:
        ids = self.tok.encode(text).ids
        if ids[-1] != 151643:  # <|endoftext|>: last-token pooling reads the EOS position, as SentenceTransformer does
            ids = ids + [151643]
        n = len(ids)
        feed = {
            "input_ids": np.array([ids], np.int64),
            "attention_mask": np.ones((1, n), np.int64),
            "position_ids": np.arange(n, dtype=np.int64)[None],
        }
        for p in self.past:
            feed[p.name] = np.zeros((1, 8, 0, 128), self.ftype)
        h = self.sess.run(None, feed)[0][0, -1, :dim].astype(np.float32)
        return h / np.linalg.norm(h)


def bm25_rank(queries: list[str], docs: list[str]) -> list[np.ndarray]:
    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    dt = [collections.Counter(tok(d)) for d in docs]
    dl = np.array([sum(t.values()) for t in dt], np.float32)
    df = collections.Counter(w for t in dt for w in t)
    N, avg = len(docs), float(dl.mean())
    out = []
    for q in queries:
        s = np.zeros(N, np.float32)
        for w in set(tok(q)):
            if w not in df:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            tf = np.array([t.get(w, 0) for t in dt], np.float32)
            s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avg))
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--variants", default="fp16,int8,q4f16")
    ap.add_argument("--queries", type=Path, default=ROOT / "scripts/search/semantic_queries.json")
    a = ap.parse_args()
    a.queries.parent.mkdir(parents=True, exist_ok=True)

    texts = card_texts()
    z = np.load(ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    codes = z["codes"].astype(np.int64)
    card = z["card"].astype(np.float32)
    eff = z["effects"].astype(np.float32)
    neff = z["n_effects"].astype(np.int64)
    # One row per card name (alt arts share text): rank names, not codes.
    names = [texts.get(int(c), ("?", ""))[0] for c in codes]
    keep = sorted({n: i for i, n in reversed(list(enumerate(names)))}.values())
    codes, card, eff, neff = codes[keep], card[keep], eff[keep], neff[keep]
    names = [names[i] for i in keep]
    row = {n: i for i, n in enumerate(names)}
    emask = np.arange(eff.shape[1])[None, :] < np.maximum(neff, 1)[:, None]
    print(f"cards {len(codes)} (unique names), effects/card mean {neff.mean():.2f}", flush=True)

    qs = make_queries(pool_cards(texts, a.n), texts, a.queries)
    targets = np.array([row[texts[q["code"]][0]] for q in qs])
    if len(qs) != 2 * a.n:
        fail(f"{len(qs)} queries, expected {2 * a.n}")

    def metrics(scores: list[np.ndarray], sel: np.ndarray) -> str:
        ranks = np.array([int((s > s[t]).sum()) for s, t in zip(scores, targets)])[sel]
        return f"R@1 {np.mean(ranks < 1):.2f} R@10 {np.mean(ranks < 10):.2f} MRR {np.mean(1 / (ranks + 1)):.3f}"

    kinds = {k: np.array([q["kind"] == k for q in qs]) for k in ("one", "two")}
    report = []

    def emit(label: str, scores: list[np.ndarray]) -> None:
        line = f"{label:34s} one: {metrics(scores, kinds['one'])} | two: {metrics(scores, kinds['two'])}"
        print(line, flush=True)
        report.append(line)

    docs = [texts[int(c)][1] + " " + names[i] for i, c in enumerate(codes)]
    s_bm25 = bm25_rank([q["q"] for q in qs], docs)
    emit("bm25", s_bm25)

    def rrf(*lists: list[np.ndarray], k: int = 60) -> list[np.ndarray]:
        out = []
        for per_q in zip(*lists):
            tot = np.zeros(len(per_q[0]), np.float32)
            for sc in per_q:
                rank = np.empty(len(sc), np.int64)
                rank[np.argsort(-sc, kind="stable")] = np.arange(len(sc))
                tot += 1.0 / (k + 1 + rank)
            out.append(tot)
        return out

    tops: dict[str, list[np.ndarray]] = {}
    for v in a.variants.split(","):
        enc = Encoder(a.onnx_dir, v)
        for instr in (False, True):
            def vec(t: str, enc: Encoder = enc, instr: bool = instr) -> np.ndarray:
                return enc((INSTRUCT if instr else "") + t)

            qv = np.stack([vec(q["q"]) for q in qs])
            if not np.isfinite(qv).all():
                fail(f"{v}: non-finite query vectors")
            s_card = list(qv @ card.T)
            s_eff = [np.where(emask, eff @ x, -1).max(1) for x in qv]
            # "X and Y": each clause's best effect, averaged, so a card must match both.
            s_split = []
            for q, base in zip(qs, s_eff):
                parts = [p.strip() for p in re.split(r"\band\b|,", q["q"]) if len(p.strip()) > 3]
                if len(parts) < 2:
                    s_split.append(base)
                    continue
                pv = np.stack([vec(p) for p in parts])
                s_split.append(np.mean([np.where(emask, eff @ x, -1).max(1) for x in pv], 0))
            tag = f"{v}{'+instr' if instr else ''}"
            emit(f"{tag} card", s_card)
            emit(f"{tag} max-effect", s_eff)
            emit(f"{tag} clause-split", s_split)
            emit(f"{tag} hybrid rrf(bm25,split)", rrf(s_bm25, s_split))
            tops[tag] = [np.argsort(-s)[:10] for s in s_eff]
        print(f"variant {v} done", flush=True)

    for tag, top in tops.items():
        ref = tops["fp16+instr" if tag.endswith("+instr") else "fp16"]
        ov = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(top, ref)])
        line = f"top-10 overlap with fp16 ({tag}): {ov:.2f}"
        print(line, flush=True)
        report.append(line)
    out = ROOT / "data/cache/search/semantic_eval.txt"
    out.write_text("\n".join(report) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
