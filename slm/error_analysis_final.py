"""
Test-set error analysis for the CHIA NER benchmark.

Headline P/R/F1 comes from the OFFICIAL scorer (chia_pipeline.eval_utils.
score_corpus_both) so numbers match score_predictions.py exactly. This module
adds only the diagnostic layer the scorer does not provide: the exact/boundary/
label/spurious breakdown, label confusions, per-type F1, and an inclusion-vs-
exclusion split.

Runs on the TEST split (held out), which is where error analysis belongs.
Auto-discovers every arm that has a test run, so once the SLM arms are tested
they appear automatically with no code change.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "src"))

import json, glob, os, re
from collections import Counter
from chia_pipeline.eval_utils import score_corpus_both

OUT = "/content/NER_repo/slm/outputs"
TEST_GOLD = "/content/NER_repo/data/processed_baseline/test_spans.jsonl"
ENTITY_TYPES = ["Person","Condition","Drug","Observation","Measurement","Procedure",
    "Device","Visit","Temporal","Value","Negation","Qualifier","Multiplier",
    "Reference_point","Mood"]

def load_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]
def _ov(a,b): return a["start"] < b["end"] and b["start"] < a["end"]

def dropped_as_fp(dr):
    out=[]
    for i,d in enumerate(dr):
        s=-2*(i+1); t=d.get("type") if d.get("type") in ENTITY_TYPES else "_invalid"
        out.append({"type":t,"start":s,"end":s+1})
    return out

def test_arms(n):
    """Every arm with a test run at shot count n. SLM arms use <model> names;
    GPT arms are labelled gpt-<tier>. Auto-includes SLM arms once tested."""
    arms={}
    for f in glob.glob(f"{OUT}/*/predictions_*_n{n}_mfull_seed42.jsonl"):
        base=os.path.basename(f)
        if "_test_" in base or "gpt-5-6" not in f:
            if "gpt-5-6" in f:
                m=re.search(r"predictions_gpt-5-6-(\w+)_test", base)
                if m: arms[f"gpt-{m.group(1)}"]=f
            else:
                arms[os.path.basename(os.path.dirname(f))]=f
    for f in glob.glob(f"{OUT}/gpt-5-6-sol/*_test_n{n}_mfull_seed42.jsonl"):
        m=re.search(r"predictions_gpt-5-6-(\w+)_test", os.path.basename(f))
        if m: arms[f"gpt-{m.group(1)}"]=f
    return arms

def _load(path, gold):
    rows={r["id"]:r for r in load_jsonl(path)}
    ids=[i for i in rows if i in gold]
    g=[gold[i]["entities"] for i in ids]
    p=[rows[i]["pred"]+dropped_as_fp(rows[i].get("dropped",[])) for i in ids]
    return ids, g, p, rows

# Headline
def headline(n=250):
    gold={r["id"]:r for r in load_jsonl(TEST_GOLD)}
    arms=test_arms(n)
    if not arms:
        print(f"No test runs at n={n} yet."); return
    print(f"TEST HEADLINE (n={n})\n")
    print(f"{'arm':16s} {'cov':>10} {'sP':>6} {'sR':>6} {'sF1':>6} | {'rP':>6} {'rR':>6} {'rF1':>6}")
    print("-"*72)
    for name,path in sorted(arms.items()):
        ids,g,p,_=_load(path,gold)
        s=score_corpus_both(g,p); e=s["exact"]["overall"]; x=s["relaxed"]["overall"]
        print(f"{name:16s} {f'{len(ids)}/{len(gold)}':>10} "
              f"{e['precision']:6.3f} {e['recall']:6.3f} {e['f1']:6.3f} | "
              f"{x['precision']:6.3f} {x['recall']:6.3f} {x['f1']:6.3f}")

# Error types
def error_breakdown(n=250):
    gold={r["id"]:r for r in load_jsonl(TEST_GOLD)}
    arms=test_arms(n)
    if not arms: print(f"No test runs at n={n}."); return
    print(f"\nERROR TYPES (n={n}): exact / boundary / label / spurious\n")
    print(f"{'arm':16s} {'exact':>7} {'bound':>7} {'label':>7} {'spur':>7}")
    print("-"*48)
    for name,path in sorted(arms.items()):
        _,g,p,_=_load(path,gold)
        k=Counter()
        for gg,pp in zip(g,p):
            matched=set()
            for pr in pp:
                hit=None; mi=None
                for gi,ge in enumerate(gg):
                    if gi in matched or not _ov(pr,ge): continue
                    ss=(pr["start"],pr["end"])==(ge["start"],ge["end"]); st=pr["type"]==ge["type"]
                    hit="exact" if ss and st else "boundary" if st else "label" if ss else "both"; mi=gi
                    if hit=="exact": break
                if hit is None: k["spurious"]+=1
                else: matched.add(mi); k[hit]+=1
        tot=sum(k.values()) or 1
        print(f"{name:16s} {k['exact']/tot:6.1%} {k['boundary']/tot:6.1%} "
              f"{k['label']/tot:6.1%} {k['spurious']/tot:6.1%}")

# Per-type
def per_type(n=250):
    gold={r["id"]:r for r in load_jsonl(TEST_GOLD)}
    arms=test_arms(n)
    if not arms: print(f"No test runs at n={n}."); return
    print(f"\n PER-TYPE strict F1 (n={n}\n")
    names=sorted(arms); tbl={}
    for name,path in arms.items():
        _,g,p,_=_load(path,gold)
        tp,fp,fn=Counter(),Counter(),Counter()
        for gg,pp in zip(g,p):
            gs={(e["type"],e["start"],e["end"]) for e in gg}
            ps={(e["type"],e["start"],e["end"]) for e in pp}
            for t,_,_ in ps&gs: tp[t]+=1
            for t,_,_ in ps-gs: fp[t]+=1
            for t,_,_ in gs-ps: fn[t]+=1
        f1={}
        for t in ENTITY_TYPES:
            pr=tp[t]/(tp[t]+fp[t]) if tp[t]+fp[t] else 0
            rc=tp[t]/(tp[t]+fn[t]) if tp[t]+fn[t] else 0
            f1[t]=2*pr*rc/(pr+rc) if pr+rc else 0
        tbl[name]=f1
    # support from gold
    sup=Counter(e["type"] for r in gold.values() for e in r["entities"])
    print(f"{'label':16s} {'support':>7} " + " ".join(f"{nm[:8]:>8}" for nm in names))
    for t in ENTITY_TYPES:
        print(f"{t:16s} {sup[t]:>7} " + " ".join(f"{tbl[nm][t]:8.3f}" for nm in names))

# Confusions
def confusions(model, n=250, top=12):
    gold={r["id"]:r for r in load_jsonl(TEST_GOLD)}
    arms=test_arms(n)
    if model not in arms:
        print("available:", sorted(arms)); return
    _,g,p,_=_load(arms[model],gold)
    conf=Counter()
    for gg,pp in zip(g,p):
        matched=set()
        for pr in pp:
            for gi,ge in enumerate(gg):
                if gi in matched or not _ov(pr,ge): continue
                if (pr["start"],pr["end"])==(ge["start"],ge["end"]) and pr["type"]!=ge["type"]:
                    conf[(ge["type"],pr["type"])]+=1
                matched.add(gi); break
    print(f"\n LABEL CONFUSIONS - {model} (gold -> predicted)")
    for (gt,pt),c in conf.most_common(top):
        print(f"  {gt:16s} -> {pt:16s} {c}")

# Inclusion/exclusion
def by_criteria_type(model, n=250):
    gold={r["id"]:r for r in load_jsonl(TEST_GOLD)}
    arms=test_arms(n)
    if model not in arms:
        print("available:", sorted(arms)); return
    rows={r["id"]:r for r in load_jsonl(arms[model])}
    print(f"\n {model}: inclusion vs. exclusion (n={n})")
    for ct in ["inclusion","exclusion"]:
        ids=[i for i in rows if i in gold and gold[i].get("criteria_type","").lower().startswith(ct[:3])]
        if not ids:
            print(f"  {ct}: no records (check criteria_type values)"); continue
        g=[gold[i]["entities"] for i in ids]
        p=[rows[i]["pred"]+dropped_as_fp(rows[i].get("dropped",[])) for i in ids]
        s=score_corpus_both(g,p)["exact"]["overall"]
        print(f"  {ct:10s} ({len(ids):4d} sents): P {s['precision']:.3f} "
              f"R {s['recall']:.3f} F1 {s['f1']:.3f}")

def full(n=250, focus="gpt-luna"):
    headline(n); error_breakdown(n); per_type(n)
    confusions(focus, n); by_criteria_type(focus, n)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Test-set error analysis for CHIA NER")
    ap.add_argument("-n", "--n-shots", type=int, default=250,
                    help="shot count to analyze (default 250)")
    ap.add_argument("--focus", default="gpt-luna",
                    help="arm to run confusions + inclusion/exclusion on")
    ap.add_argument("--section", default="all",
                    choices=["all", "headline", "errors", "per-type",
                             "confusions", "criteria"],
                    help="which analysis to run")
    args = ap.parse_args()

    if args.section == "all":
        full(args.n_shots, focus=args.focus)
    elif args.section == "headline":
        headline(args.n_shots)
    elif args.section == "errors":
        error_breakdown(args.n_shots)
    elif args.section == "per-type":
        per_type(args.n_shots)
    elif args.section == "confusions":
        confusions(args.focus, args.n_shots)
    elif args.section == "criteria":
        by_criteria_type(args.focus, args.n_shots)


if __name__ == "__main__":
    main()