# -*- coding: utf-8 -*-
"""eval DPO_v3: 加载 base+merge(v3 SFT)+DPO adapter, 60 holdout 生成, 对比 v3 SFT。
指标: 词长/坍缩/behaviour/开放提问率。推理设置同 eval_bcw。"""
import os, json, argparse, numpy as np, torch

ap=argparse.ArgumentParser()
ap.add_argument("--dpo", default="outputs/dpo/qwen14b_dpo_v3")
ap.add_argument("--out", default="data/annomi/responses_dpo_v3.jsonl")
_a=ap.parse_args()

BASE="/root/autodl-tmp/models/Qwen3-14B"
V3="outputs/mm_sft/qwen14b_sft_2048_bcw3"
DPO=_a.dpo
HOLD="data/annomi/mm_sft_final/holdout.jsonl"
OLD="data/annomi/responses_14b_bcw3.jsonl"   # 有 v3 SFT(sft_bcw)
OUT=_a.out
N=60

def nw(s): return len((s or "").split())

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
dev="cuda"
tok=AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
rows=[json.loads(l) for l in open(HOLD,encoding="utf-8")][:N]
old=[json.loads(l) for l in open(OLD,encoding="utf-8")][:N]

print("[load] base + v3 SFT (merge) + DPO adapter")
lm=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(dev)
sft=os.path.join(V3,"lora_adapter"); lm=PeftModel.from_pretrained(lm, sft if os.path.isdir(sft) else V3)
lm=lm.merge_and_unload()
lm=PeftModel.from_pretrained(lm, DPO).eval()

@torch.no_grad()
def gen(msgs):
    try: txt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: txt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ids=tok(txt,return_tensors="pt",truncation=True,max_length=1536).input_ids.to(dev)
    torch.manual_seed(0)
    o=lm.generate(ids,max_new_tokens=96,do_sample=True,temperature=0.6,top_p=0.9,
                  repetition_penalty=1.05,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip().split("</think>")[-1].strip()

dpo_out=[]
for i,r in enumerate(rows):
    dpo_out.append(gen(r["messages"][:-1]))
    if (i+1)%15==0: print(f"  gen {i+1}/{N}",flush=True)

with open(OUT,"w",encoding="utf-8") as f:
    for i,r in enumerate(rows):
        f.write(json.dumps({"user_text":old[i].get("user_text",""),"gold":old[i].get("gold",""),
                            "sft_v3":old[i].get("sft_bcw",""),"dpo_v3":dpo_out[i]},ensure_ascii=False)+"\n")

# 行为 + 开放提问
import sys; sys.path.insert(0,"scripts")
def behdist(texts):
    try:
        import behaviour_scorer as BS
        labs,_=BS.predict(texts,"cpu"); n=len(labs)
        return {k:round(100*sum(1 for l in labs if l==k)/n) for k in BS.KEYS}, labs
    except Exception as e:
        print("beh skip:",e); return {},[None]*len(texts)

d=np.load("outputs/evaluator/oc_clf.npz",allow_pickle=True)
coef,intercept,keys=d["coef"],d["intercept"],[str(k) for k in d["keys"]]
from sentence_transformers import SentenceTransformer
M=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")
def open_rate(texts, labs):
    qidx=[i for i,l in enumerate(labs) if l=="question"]
    if not qidx: return 0,0
    E=np.asarray(M.encode([texts[i] for i in qidx],show_progress_bar=False),dtype=np.float32)
    pred=(E@coef.T+intercept).argmax(1)
    opens=sum(1 for p in pred if keys[p]=="open")
    return opens,len(qidx)

sft_v3=[old[i].get("sft_bcw","") for i in range(N)]
print("\n"+"="*60+"\nDPO_v3 vs SFT_v3 (60 holdout)\n"+"="*60)
for name,texts in (("SFT_v3",sft_v3),("DPO_v3",dpo_out)):
    wl=np.mean([nw(t) for t in texts]); bc=100*np.mean([nw(t)<=3 for t in texts])
    bd,labs=behdist(texts)
    op,nq=open_rate(texts,labs)
    print(f"  {name}: 词长{wl:.1f} 坍缩{bc:.0f}%  {bd}  开放问 {op}/{nq}={100*op/max(1,nq):.0f}%")
print("\n样例:")
for i in range(min(6,N)):
    print(f"  IN:{old[i].get('user_text','')[:55]}")
    print(f"   SFT:{sft_v3[i][:70]}\n   DPO:{dpo_out[i][:70]}\n")
print("wrote ->",OUT)
