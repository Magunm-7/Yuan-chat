# Yuan-chat / MPSE — 多模态状态感知的动机式访谈(MI)咨询对话模型

> 一个**完整、可调用**的两段式系统:用语音语调 + 面部表情 + 文本**感知来访者的心理状态**(改变意愿 / 唤醒 / 效价),据此**生成更共情、更贴状态的咨询回复**。全程**无人工标注**,弱监督自建信号;数据用公开基准 **AnnoMI**。
>
> 定位:工程/研究导向的**完整闭环**(方法 + 能跑的系统 + 有对照的结果 + 诚实的局限),不追求虚的 SOTA。

---

## 0. TL;DR — 这个项目做成了什么

- **A 段·评估器(感知层)**:一个 0.89M 的小网络,吃三模态(Whisper 语音 / CLIP 面部 / MiniLM 文本),一次前向输出每句话的 **μ(状态均值)/ σ(不确定性)/ α(模态门控权重)**。整体消融证明多模态感知**有效**:质量判别 AUC **0.717 > 0.638**(不过评估器)。
- **B 段·生成器(表达层)**:冻结 **Qwen3-14B** + LoRA,吃 40 轮对话历史 + 评估器状态标记,生成 MI 咨询回复。
- **本轮核心成果(reward shaping + DPO)**:从"生成器坍缩成 'Mm-hmm' 附和"这个真实失败出发,
  1. 回头审数据 → 定位坍缩根因是训练目标里 **41% 是 backchannel(附和语)** → 用**行为子类 + 会话质量精细加权**修复(SFT_v3,坍缩 92%→37%,真坍缩 88%→15%);
  2. **用来访者的真实反应(talk_type 转变)而非开发者品味来定 reward** → 发现"开放式提问"是唯一能把来访者推向"改变"的强信号(净 Δ +0.112) → 据此设计离散 reward;
  3. DPO 优化时撞上 **reward hacking**(模型把"开放提问"钻成 "So...?" 万能句刷屏)→ 用 **beta 剂量-反应曲线**证明病根是"过度优化"、收紧隐式 KL 修复;
  4. **独立 GPT-4o judge 确认:最终产物 DPO_v3(beta=0.5)> SFT_v3 > 真人专家 gold**。
- 全链**诊断 → 修复 → 独立验证**闭环,每一步都有量化 + 诚实边界。

---

## 1. 问题与数据

**痛点**:数字心理健康 / 情感陪伴 AI 是真实在增长的赛道(Woebot / Wysa 等),但现有多为**纯文本**,读不懂来访者"怎么说"(语气、表情)。我们做多模态状态感知的共情咨询对话。

**数据:AnnoMI**(公开基准,ICASSP 2022)。133 段专家标注的 MI 访谈(110 高质量 / 23 低质量),含逐句的:
- `client_talk_type` ∈ {change, neutral, sustain} —— 来访者是朝"改变"还是"维持现状"说话(**结果信号**);
- `main_therapist_behaviour` + 细粒度子类(reflection: simple/complex;question: open/closed;input: information/advice/negotiation/options)——咨询师做了什么动作;
- `mi_quality`(会话级 high/low)。

清洗后 **128 session / train 2433 / holdout 491**(会话级划分不泄漏),每条带 **40 轮对话历史**。

---

## 2. A 段 · 评估器 MPSE(感知层)

### 三个核心 idea
1. **μ / σ / α**:异方差回归一次前向同时输出状态均值、不确定性、模态门控权重。
2. **用状态轨迹找信号**:不做逐句因果归因(数据证明其为空),而是读**会话尺度的状态轨迹形状**区分咨询质量。
3. **用 σ 反哺训练**:样本权重 `w = exp(-λ·σ̄)`,越不确定权重越低。

### 架构
`三模态编码器(冻结:Whisper768 / CLIP768 / MiniLM384) → 投影 → α 门控加权和 → GRU 时序 → μ/σ 异方差头`。

### 结果(session-CV,袋外预测)
- **整体消融(证明"评估器加了信息")**:质量判别 Option C AUC **full 0.717 > no-eval 0.638**;残缺的纯文本版 0.533 反而更差 → **多模态融合是刚需**。
- 四开关消融:去视频 −0.064(面部最重要)、去音频 −0.037、去 α −0.028、**去 σ +0.005(σ 是这份数据上最弱的一环,如实报告)**。
- H1:μ_chg 逐句还原 talk_type,AUC 0.615 / p=0.001。
- **诚实边界**:效应量多在 std 量级(23 个 low session 是功效硬上限);σ 在质量判别上未显增益。

---

## 3. B 段 · 生成器(表达层)

**基座**:冻结 **Qwen3-14B**(不量化)+ LoRA(r=8, α=16, dropout=0.05, 挂 q/k/v/o)。
**输入**:40 轮对话历史 + 当前来访者话 + 评估器 μ 状态标记(`[Observed client state — change-readiness: …]`)。
**配置**:text_only(音视频经评估器→μ→标记进 prompt,不直接进生成器)、max_len 2048、bf16、gradient checkpointing。

> 上下文收益边界(独立发现):ppl 在 1536 轮历史处已饱和(1536→2048 仅 Δ0.001),14B 受显存吃满 2048 并不构成劣势。

### 3.1 发现并修复"坍缩"

低温采样下,原始 SFT 会坍缩成 "Mm-hmm./Yeah." 附和(holdout 上 92% 输出 ≤3 词)。**回头审数据发现根因**:训练目标(咨询师回复)里 **41% 本身就是 backchannel**(光 "Mm-hmm." 占 10%)——模型忠实地学了这个众数。

**修复 = 精细样本加权**(不删数据,只降权;用 AnnoMI-full 的**金标准子类**按 utterance join,SFT 侧零分类噪声):

| 目标动作 | 权重 | 依据 |
|---|---|---|
| backchannel(≤3词) | ×0.01 | 附和是数据众数,机器也做不到真人的即时附和 |
| 复杂反映 / 开放提问 | ×1.3 | MI 高阶技术 |
| 协商 | ×1.2 | 引出改变计划 |
| 封闭提问 / 灌输 | ×0.6 | 低产出 |
| 给建议(矫正反射) | ×0.4 | 数据实证:advice 后来访者最容易转向"维持现状" |
| 低质量会话整段 | ×0.4 | 那 23 段是"坏 MI"示范 |

**结果(SFT_v3 = `qwen14b_sft_2048_bcw3`)**:整体坍缩 92%→**37%**;拆开看,**真坍缩(该给实质反映却说附和)从 88% 降到 15%**——其余 37% 是"该短就短"(gold 本身也有 43% 是最小鼓励)。行为分布:提问 38%、反映上升、附和大幅下降。

### 3.2 用"来访者结果"定 reward,做 DPO

**尺子由谁定?—— 让来访者的真实反应定,而不是开发者品味。** 分析全部 1496 处 `client_talk_type` 相邻转变,看**哪种咨询师动作之后来访者转向"改变"**:

| 动作 | 净 Δ(朝改变移动) | 下一句 change% |
|---|---|---|
| **开放提问** | **+0.112(全场最高)** | 38.5% |
| 协商 | +0.072 | 48% |
| 复杂/简单反映 | ≈ 0 | ~25% |
| 封闭提问 | −0.027 | 17.7% |
| backchannel | −0.036 | — |

→ **开放提问是唯一稳的"改变驱动"信号**(反映的价值在会话弧线,单句测不出,故保留但不靠它)。据此的**纯离散 reward**(离散类别才 DPO-safe,连续信号会被 hack):
```
{ reflection: 1.0, open-question: 1.0, closed-question: 0.1, other: 0.2, therapist_input: -0.5 }
```
open/closed 用一个 MiniLM+LogReg 分类器判(CV 0.75;实测启发式正则只有 68.5%,因为 MI 的 open 是功能性而非句法的)。

**DPO**:on-policy 8 选 1(候选来自 SFT_v3 自己,gold/base 不当榜样否则退化成 SFT);参考模型 = SFT_v3;`loss = -logσ(beta·[(logπ_pol−logπ_ref)_chosen − (…)_rejected])`。

### 3.3 reward hacking 与 beta 修复(方法论亮点)

beta=0.1 时 DPO 把"开放提问"这个**平顶** reward 钻成漏洞:95% 输出以 "So..." 开头,"So you're feeling what?" 一字不差刷了 6 次——**指标全中,质量崩坏**(典型 Goodhart)。收紧隐式 KL(beta)后单调修复:

| | 坍缩 | 'So' 开头 | 前2词多样性 |
|---|---|---|---|
| SFT_v3 | 37% | 0% | 37% |
| DPO beta=0.1 | 2% | **95%(刷屏)** | 25% |
| DPO beta=0.3 | 3% | 65% | 47% |
| **DPO beta=0.5(最终)** | 5% | **45%** | **55%** |

这条剂量-反应曲线本身**证明病根是"过度优化"而非 reward 本身烂**。

### 3.4 独立 judge 验证(GPT-4o 成对盲评)

| 对比 | 结果 |
|---|---|
| **DPO(b0.5) vs SFT_v3** | DPO 45% / SFT 35% / 平 20 → **RL 净提升 SFT** |
| **DPO(b0.5) vs 真人 gold** | **DPO 60% / gold 28%** |
| DPO(b0.3) vs DPO(b0.5) | 打平(13:12,平35) |

**排名:DPO_v3 > SFT_v3 > gold。**

> **诚实边界(必读)**:这把 judge 有**冗长偏好**——未微调的 base(23 词话痨)能打真人 gold 打到 **95%**,说明"赢"掺了"更长更讨喜"。我们只在**词长同量级**(SFT/DPO/gold 都 7–14 词)的比较里信它。"赢过 gold"也含"我们给实质回复、而 gold 那批有不少极简附和"的成分,**不等于"比人类咨询师强"**。识别并量化这个 judge 偏见,本身是本项目的一个方法论收获(它推翻了早期"跑不过 base"的表面结论)。

---

## 4. 展示样例(最终产物 DPO_v3_b05,holdout)

来访者话 → 咨询师回复:

| 来访者 | DPO_v3_b05 回复 |
|---|---|
| "He's a good flatmate."(谈室友) | **"He's a good flatmate. What do you like about him?"**(反映 + 开放提问) |
| "It's normal for university students I feel."(为喝酒辩护) | **"Do you think it's normal for university students to drink?"**(把判断权还给来访者) |
| 谈起想戒烟的材料 | **"Tell me a bit more about why you don't want to quit."**(引出式开放提问) |
| "I wanna be able to look nice." | **"So, what does looking nice mean to you?"**(探索价值) |

**方法论对照(同一条输入,三个阶段)**——展示"坍缩 → hacking → 修复":

| 阶段 | 对 "我昨晚 pub crawl 摔伤了脚踝…" 的回复 |
|---|---|
| 原始 SFT(坍缩) | "Mm-hmm." |
| DPO beta=0.1(hacking) | "So you're feeling what?"(空洞万能句,刷屏) |
| **DPO beta=0.5(最终)** | "What happened? What went wrong?"(切题开放提问) |

---

## 5. 诚实的局限

- 评估器的 σ 反哺在这份数据上未显增益;效应量受 23 个 low session 功效上限限制。
- 音视频对**生成**几乎无增益(三种注入 × 两种训练量全试过)——音视频的价值在**感知层**(评估器质量判别),不在生成;这与 A 段结论一致。
- LLM-as-judge 有冗长偏见(见 §3.4),所有"胜率"需扣着这个读。
- 最终产物残留一个 "So..." 风格 tic(45%,多为合法反映式反问);DPO 相对 SFT 的领先是温和的(+10 点),非碾压。
- AnnoMI 是**演示性质**的专家访谈,非真实临床;不声称跨语言/临床有效性。

---

## 6. 关键产物与文件

```
outputs/
  evaluator/  mpse_deploy.pt      # 可实时推理的评估器
              behaviour_clf.npz   # 4类行为分类器(reward用)
              oc_clf.npz          # open/closed 分类器(reward用, CV 0.75)
  mm_sft/     qwen14b_sft_2048_bcw3/   # ★ SFT_v3(精细加权,治坍缩)
  dpo/        qwen14b_dpo_v3_b05/      # ★★ 最终产物(beta=0.5)
              qwen14b_dpo_v3_b03/      #    beta=0.3(与b05打平)
data/annomi/
  turns_labeled.jsonl             # 逐 client turn(gold标签 + 弱标签)
  mm_sft_final/{train,holdout}.jsonl   # SFT 训练/评估集(40轮历史+状态标记)
  cand_pool_v3.jsonl / pairs_v3.jsonl  # DPO 候选池 / 偏好对
  talk_type_transitions.md        # 1496 处 talk_type 转变全量分析
  compare_sft_dpo_v3.md           # SFT vs DPO 逐条对比
  AnnoMI-full.csv                 # 原始数据(含行为子类)
scripts/                          # 全套管线脚本(见 run_scripts/ 内的一键脚本)
notes.md                          # 完整研发日志(最权威的过程记录)
```

**推理**(最终产物):`base Qwen3-14B → merge(SFT_v3 LoRA) → 挂 DPO_v3_b05 adapter`;状态标记已烘进 prompt,推理走纯文本路径。

**环境**:14B 用 `qwen3` env(transformers 4.51.3);评估器/reward 相关用 base env。数据/产物在 `data/annomi/`、`outputs/`(gitignore,只在服务器 + 本地备份,不在 GitHub)。

---

## 7. 完整方法论一句话

> 察觉生成器坍缩 → 回头审数据定位根因(backchannel 噪声)→ 精细加权修复(SFT_v3)→ 诊断"跑不过 base"是 judge 冗长偏见 → 用来访者结果信号(talk_type)定 reward → DPO 撞 reward hacking → beta 剂量-反应证明过度优化并修复 → 独立 judge 确认 DPO_v3 > SFT_v3 > 真人 gold。**从真实失败一步步逼出来,不是照论文抄。**
