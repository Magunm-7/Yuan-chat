# Yuan-chat / MPSE · 项目进度笔记

> 最后更新：2026-07-22
> 用途：新开对话时读这份，能恢复到"已经把整个项目和代码吃透"的状态。

---

## 0. 当前工作范围（⚠️ 先看这条）

**只做框架和方法。数据另说。**

- ✅ **在范围内**：架构评估（第 6.5 节）指出的问题、评估方法怎么设计、代码工程健康度（路径、可复现性、README）。
- ❌ **不在范围内**：现有 19 段 / 371 turn 数据的任何修补（补 therapist_reply、清 npz 残留、重训）。这批是弃用数据，**别再被它带走注意力**。
- ⏸ **挂起中的上游决定**：真实数据到底用哪个（AnnoMI 主数据？DAIC-WOZ / CMDC 做带标签外部验证？）。用户会在另外的时间点定，**不要催、不要替他定**。

第 5 节的数据统计、第 6 节里标了「数据相关」的条目，都只作为**历史记录**保留，不是待办。

**目标**：不是论文，不是 SOTA。是**有始有终把故事说完整** —— 让项目从"一条能跑通的 pipeline"变成"一个有结论的实验"。

---

## 0.1 上下文来源说明

原始那段深聊（对话标题 "Yuan-chat"，2026-06-11）**已经被 Claude Code 自动清理掉了**（transcript 默认只留 30 天）。
本笔记重建自三个来源：

1. `C:\Users\qmn20\claude-yuanchat-notes.md` —— 当时手动落盘的存档笔记。内容已 100% 并入本文件，**原文件已于 2026-07-22 删除**。
2. 会话 `2d026650-f43b-4c4f-adf6-886e3457d47b`（2026-07-09 11:04–13:10）—— 那次为了改简历，重新读了一遍旧对话 + 通读了全部代码
3. 直接读代码和数据（本次，2026-07-22）

**教训：重要结论要落盘成文件，不要指望对话记录还在。**

---

## 1. 协作方式（用户偏好，先看）

- 用户（孟楠）吃**直接、不和稀泥的校准**，该说 no 就说 no，别给安慰剂。
- 两个反复出现的老毛病，advising 时盯着：
  - **好高骛远**：习惯把目标定在最难那一档。
  - **all-or-nothing 的自我否定 / 回避**："这坨太烂不如全扔"、"等等再说"。要把**精准截肢（健康）**和**连好的一起烧（自毁）**分开。
- 也容易**妄自菲薄**：把手里的真牌（多模态算法实习、能跑通的项目）划掉说"我没有"。
- **不要用脚本整份重生成用户手改过的文件**（曾经差点覆盖他手改的简历）；改之前先读当前文件，就地改。

---

## 2. 项目是什么

**一句话**：输入一段心理访谈视频 → 自动切 turn → 三路弱监督**自动造心理状态标签** → 训小网络 MPSE 预测每句的 μ(状态均值)/σ(不确定性)/α(模态证据权重) → 用 μ/σ 给样本算权重 → 音视频前缀 + LoRA 微调 LLM 生成"咨询师回复"。

两段式：**A 段 = 评估器（MPSE）**，**B 段 = 生成器（多模态前缀 SFT）**。

### 三个核心 idea（项目灵魂，真正属于用户自己的东西）

1. **μ / σ / α**：异方差回归一次前向同时输出状态均值、不确定性、模态门控权重。
2. **用 Δμ 找"有效回复"**：连续两个 user turn 之间状态下降（Δμ<0）即认为上一条咨询师回复有效。
3. **用 σ 反哺训练**：样本权重 `w = exp(-λ·σ̄)`，越不确定权重越低，自动降权低置信样本。

### 8 阶段闭环

| # | 阶段 | 产物 |
|---|------|------|
| 0 | ffmpeg 抽 16k 单声道音频 | `data/derived/<sid>/audio_16k.wav` |
| 1 | 能量 VAD 切 turn + faster-whisper 转写 + 三路弱打分融合 | `turns.jsonl` |
| 2 | 造 MPSE 训练集（X=文本 emb + 4 标量, Y=y_soft, Q=质量权重） | `outputs/mpse/<sid>/train.npz` |
| 3 | 训 MPSE（MLP + 质量加权异方差高斯 NLL + α 熵正则） | `mpse.pt` / `meta.json` |
| 4 | upgrade：写 μ/σ/α/weight/effective/p_ok | `outputs/upgrade/<sid>/turns_upgraded.jsonl` |
| 5 | 造 SFT 目标（**现在靠人工填 `therapist_reply`**） | `outputs/sft/<sid>/sft_train.jsonl` |
| 6 | Whisper/CLIP 编码音视频，缓存 per-turn 特征 | `outputs/mm_cache/<sid>/turn_*.npz` + `mm_index.jsonl` |
| 7 | 多模态前缀 + LoRA SFT | `outputs/mm_sft/<sid>/mm_prefix.pt` + `lora_adapter/` |

---

## 3. ⚠️ 代码有两份，git 仓库那份是过时的

| | `C:\Users\qmn20\Yuan-chat`（git 仓库） | `C:\Users\qmn20\Desktop\Yuan-chat\待打包材料\Yuan-chat` |
|---|---|---|
| 代码时间 | 2026-02-03（commit `251d4f8 daily modify`） | 最新（README 描述的就是这份） |
| 数据 | 只有 S0001，16 turns | **S0001–S0019，371 turns** |
| 训练产物 | 无 | `mm_sft/ALL`（LoRA + mm_prefix.pt，已训完） |
| 版本控制 | 有 `.git`，remote = github.com/Magunm-7/Yuan-chat | **无 git** |
| 嵌套 | 真代码在子目录 `mpse_full_loop_project/` | 直接平铺 |

**git 仓库的 README.md 是 2026-02-25 单独更新的，描述的是桌面那份新代码**，所以 README 和同仓库的代码对不上（README 讲 pipeline 开关、`merge_mm_index.py`、Qwen3-8B，仓库代码里全都没有）。这是当初"README 说 Qwen3-8B 但 config 是 Llama-3.2-3B"那个矛盾的真正原因——不是笔误，是**代码没推上去**。

### 新版相比 git 版多了什么

- `configs/default.yaml`：新增 `pipeline:` 分阶段开关 + `upgraded_policy: skip_if_exists`（保护手改的 `turns_upgraded.jsonl`）、`dialogue:` 段；模型从 Llama-3.2-3B 换成 **Qwen3-8B**；`session.id: "ALL"`
- `scripts/run_full_loop.py`：7128 → 15328 字节，加了分阶段开关、`--session_id` 覆盖、train-only 快捷路径、缺文件的显式报错
- `scripts/merge_mm_index.py`（新）：把 19 个 session 的 mm_index 合并成 `outputs/mm_cache/ALL/mm_index.jsonl`
- `src/mpse_mvp/segment/diarize.py`（新）：MFCC + 手写 kmeans(k=2) 的轻量说话人分离 —— **写了但没接进 build_turns.py，目前是死代码**
- `src/mpse_mvp/sft/build_sft_from_pairs.py`（新）：dialogue 模式下用真实 assistant 回复造 SFT
- `src/mpse_mvp/pipeline/build_turns.py`：5543 → 11462 字节，加了 dialogue 模式（交替分配 role）、`force_single_turn`

### 新版已经修掉的旧 bug

- ✅ **SFT 目标模板污染**：旧 `teacher_generate.py` 用 `split("ASSISTANT:")[-1]` 抽不出 Llama chat 模板，整个 prompt 被当训练目标。新版 `teacher.enabled=false`，改走人工 `therapist_reply`，绕开了。
- ✅ **label 没做 mask**：旧 `data_mm.py` 里 `labels = input_ids.clone()`，system+user 也在算 loss。新版 `_build_prompt_and_full()` 正确把 prompt 部分置 -100，**只训 assistant token**。
- ✅ **sample_weight 用错**：旧 model_wrap 拿 batch 里权重的均值去缩放整个 loss。新版改成 per-token CE + per-sample 归一 + 逐样本加权。
- ✅ **α 的 video 键拿错**：旧 cache_builder `v = alpha_dict.get("audio", ...)`（复制粘贴错误，video 权重取到了 audio）。新版已 BUGFIX。
- ✅ **LoRA 被冻住**：新版 freeze 逻辑认 `.lora_A/.lora_B`，并显式解冻 projector / mu_head。

---

## 4. 代码地图（以桌面新版为准）

```
configs/default.yaml              全局配置：pipeline 开关 / VAD / ASR / 弱监督 / MPSE / upgrade / mm / mm_sft
scripts/
  run_full_loop.py                主编排，7 阶段各有开关；--session_id 可覆盖
  merge_mm_index.py               合并各 session 的 mm_index -> ALL
  demo_mm.py                      推理 demo：加载 LoRA + mm_prefix，前 5 条生成回复
  download_models_ms.py           ModelScope 下模型
src/mpse_mvp/
  segment/
    extract_audio.py              ffmpeg mp4 -> 16k 单声道 wav
    vad.py                        能量 VAD（分帧 RMS > thr），非重叠帧
    diarize.py                    ★ MFCC + kmeans(k=2) 说话人分离 —— 未接入
    io.py                         soundfile 读 wav
  asr/whisper_asr.py              faster-whisper 整段转写 + 按时间窗 gather_text
  features/
    text_features.py              q_text = clip(len/40); 4 维简单文本特征（长度/否定词/问号/叹号）
    audio_features.py             q_audio = RMS/MAD 的 SNR 代理; stress_proxy = 谱质心归一化
    video_features.py             MediaPipe FaceLandmarker -> q_video(人脸可见率) + microexpr_rate(关键点位移/眼距)
  supervision/
    agents.py                     三路弱打分器 + 按质量加权融合 fuse_labels
    llm_rater.py                  LLM 按临床 rubric 输出 {dep,sad,anx,stress} 严格 JSON
  pipeline/
    build_turns.py                VAD -> (可选 diarize/dialogue) -> 三路打标 -> turns.jsonl
    build_mpse_trainset.py        X = 文本 emb(mean-pool) ⊕ [q_text,q_audio,q_video,microexpr]; Y=y_soft; Q=质量
  mpse/
    model.py                      2 层 MLP -> 三头: α(softmax over 3) / μ(sigmoid) / logvar(clamp -6..2)
    train.py                      loss = 质量加权高斯 NLL + 0.01·(-α 熵)
  upgrade/upgrade.py              推理写 μ/σ/α; w=exp(-λσ̄); Δμ<0 -> effective_raw; σ<σ_max 才 trusted; p_ok=Φ((τ-μ)/σ)
  sft/
    build_sft.py                  从 turns_upgraded 造 SFT，assistant 取人工填的 therapist_reply
    build_sft_from_pairs.py       dialogue 模式：assistant 取 pairs.jsonl 里的真实回复
    teacher_generate.py           自蒸馏造回复（已弃用，teacher.enabled=false）
  mm/
    encoders.py                   WhisperAudioEncoder(encoder mean-pool) / CLIPVideoEncoder(CLS over frames)
    cache_builder.py              per-turn 存 audio_feat/video_feat/alpha(2维)/mu
    projector.py                  SoftTokenProjector: (B,C) -> (B,K,D) 的 K 个软 token
    model_wrap.py                 MultiModalPrefixLM: 音/视频软 token 拼在 embedding 前，α 缩放，prefix 位置 label=-100
    data_mm.py                    ★ 只训 assistant token 的 label mask + collate
    train_mm_sft.py               冻结基座 + LoRA(q/k/v/o_proj, r=8) + 训 projector，存 mm_prefix.pt & lora_adapter
```

### 关键超参
- `indices.names = [dep, sad, anx, stress, microexpr_rate]`，τ 全 0.30
- `upgrade.sigma_lambda = 2.0`，`sigma_max = 0.12`
- `mm.n_frames = 8`，`k_audio = k_video = 8`
- `mpse: epochs=5, bs=8, lr=3e-4, hidden=256, dropout=0.1`
- `mm_sft: bs=1, lr=2e-4, epochs=1, max_len=1024`；LoRA r=8, alpha=16, dropout=0.05

---

## 5. 真实进度（★ 推翻旧存档笔记的结论）

旧存档笔记（6-11）说"标签几乎是常数、`effective_trusted` 全场为 0、Δμ 灵魂命题没演示出来"——**那是基于 git 仓库那份 2 月的旧 S0001 数据**。新版数据已经不是这样了：

| 指标 | 旧（git repo S0001） | 新（桌面 19 sessions） |
|---|---|---|
| session 数 | 1 | 19 |
| user turn 数 | 16 | **371** |
| y_soft 动态范围 | dep 全在 0.24–0.29（≈常数） | **0.15 – 0.60**（有真实区分度） |
| σ 范围 | 0.07 – 0.71（太大） | **0.05 – 0.44**（多数 <0.12） |
| effective_raw | 7 | **182** |
| effective_trusted | **0** | **133** |
| α | 近乎常数（std 0.03） | **std 0.08–0.10，范围 0.01–0.77** |

**所以"用 Δμ 找有效回复"这个灵魂命题现在是演示出来了的**，α 也有了真实区分度。这是项目最大的进展，之前的自我评价（包括简历里那句"暂无量化结果"）已经过时。

### 已完成
- 19 段访谈全部跑完 0–6 阶段
- 371 个 user turn 中 **268 个已人工补上 `therapist_reply`**（对照 `可用对话/*.pdf` 转录稿抄的）
- `mm_cache/ALL/mm_index.jsonl` 已合并（371 条）
- **MM-SFT 已训完**：`outputs/mm_sft/ALL/` 有 `mm_prefix.pt`(138MB) + `lora_adapter/`(31MB)

---

## 5.5 已定下的方向性决策（源自 6-11 存档，仍然有效）

**关于数据**
- **没有人工标签不是问题**：标签 100% 自动生成（LLM rater + 三路启发式），任何视频丢进来都能跑。真问题一直是**标签太弱**，不是没标签。
- **S0001–S0019 这批数据当初就已经决定弃用**：那是用户自己演的烟雾测试数据，真正切分好的留在原单位，没传 GitHub。
  → 这跟 2026-07-22 用户说的"无需在意目前的训练集，可以当没有训练集"是**同一个决定**，前后一致。所以现在的任务确实是「架构评估 + 补验证方法」，不是「把这 371 条调好」。
- **走英文 + 保三模态（含视频）**。理由：公开的多模态心理咨询视频，好货基本都在英文（AnnoMI 那一类）；英文的 ASR 和 LLM-rater 质量都更高；非文本模态本身语言无关。
  诚实边界：**不声称跨语言临床有效性**。
- **中英差别远不止翻译**。除了 tokenization、ASR/LLM 质量不对称之外，最深的一层是「心理痛苦的表达是文化绑定的」（中文更躯体化），**构念本身不跨语言**。这是上面选英文的真正理由，不是图省事。

**关于切分**
- 切分要用 **speaker diarization（pyannote / WhisperX）**，不是"很吊的多模态方案"。
  → 对照现状：`segment/diarize.py` 里是自己手写的 MFCC + kmeans(k=2)，且**没接进 build_turns**。当前实际走的是"人工剪掉机器声音 + 能量 VAD"。接 pyannote 是这条决策的落地动作。

**还没解的**
- 真实数据具体用哪一个：AnnoMI 做主数据？DAIC-WOZ / CMDC 做带标签的外部验证（正好补第 6.5 节 A1 那个循环论证的窟窿）？**这是上游决定，定了它，切分方式和模态配置才有答案。**

**用户画像（advising 时的背景）**
- 双非本科 + UNSW AI 方向硕士（2027.01 毕业），英语不差。
- 两段实习：① 研究院多模态算法（真算法经历，无论文）；② 智驾公司 AI 技术支持（写 skill / 内部 AI 工具推广，已离职，觉得不构成护城河）。
- 无顶会论文、无竞赛排名、无自动驾驶领域算法经历。
- **秋招约 2026 年 8 月开始**，学业也没完全跟上。

---

## 6. 当前已知问题

### 在范围内 —— 方法与工程

1. **完全没有 eval**（最大的一块）。没有 hold-out、没有任何指标、没有消融、没有 baseline。
   这是"讲成一个能拿出手的项目"最后一道坎，详见第 6.5 节末尾的七件套清单。
2. **绝对路径写死**：`configs/default.yaml` 里 6 处模型路径 + `features/video_features.py` 里 `face_landmarker.task` 的路径。
   → 建议加 `paths.models_root`，其余用相对路径拼。别人 clone 下来才跑得动。
3. **`merge_mm_index.py` 写死绝对路径**（`make_abs=True` 直接 `resolve()`），导致 index 跟机器绑定，换机器全废。应该存相对路径，运行时再拼。
4. **README 严重过期**。现在这份写于 2026-02-25，描述的流程和结果口径都要重写。这是项目"讲完整"的门面，放到最后做。
5. **口径要修正（零代码成本）**：σ 讲成 noise estimate 而非"模型不确定性"；Δμ 讲成 proxy signal 而非因果有效性判定。详见 6.5 的 A3 / A4。
6. **MPSE 这一级"多模态"名不副实** → 见 6.5 的 A2。
7. **`diarize.py` 写完没接线**，dialogue 模式用的是"VAD 段严格交替分配 role"这种脆弱假设 → 落地动作是接 pyannote，见 5.5。
8. **`microexpr_rate` 容易饱和**：`clip(m / 0.015, 0, 1)` 阈值定得太低，旧数据里直接常数 1.0。换数据后要重新标定这个除数。

### 已解决（2026-07-22）
- ~~代码没推 GitHub、仓库停在 2026-02-03~~ → 已把最新代码搬进仓库、拍平嵌套目录，分支 `sync-latest-code`
- ~~目录 cruft：`mpse_full_loop_project/`、`data & output/`~~ → 已删，并加了 `.gitignore`

### 数据相关 —— ⚠️ 不在当前范围，仅作记录
> 以下全部属于弃用数据的产物，**不要去修**。换数据后这些问题自然消失，或需重新评估。
- 102/371 个训练样本的 target 是字面量 `(FILL_THERAPIST_REPLY_HERE)`（`build_sft.py` 在 reply 为空时填占位符，下游没过滤）。
  → 但这**暴露了一个真实的代码缺陷**：`build_sft.py` 缺 `skip_if_no_pair` 那样的过滤开关，而 `build_sft_from_pairs.py` 里有。换数据后同样会踩，属于范围内要修的。
- `mm_cache/ALL/mm_index.jsonl` 里 npz_path 是 Linux 绝对路径 `/home/qmn/...`，Windows 上读不到（同上，根因是第 3 条）。
- `mm_cache/S0003` 有 48 个 npz 但 index 只有 12 条（旧运行残留）；`mm_cache/S0009` npz=23 但 index=24，少一个会抛异常。

---

## 6.5 方法架构评估（2026-07-22）

> 前提：**先不看现有训练集**，只评方法本身是否立得住。
> 结论先行：**架构骨架是站得住的，三个核心 idea 也是真的。缺的不是想法，是"闭环验证"那一环。**
> 下面 9 条按"会不会被内行一戳就穿"排序。

### A 段（MPSE 评估器）

**A1. 循环论证 —— 最根本的一条**
MPSE 的训练目标 Y 就是三路启发式融合出来的 `y_soft`。所以 MPSE 学到的是「拟合这三条启发式规则的加权平均」，μ 本质上是「文本关键词计数 + 谱质心 + 人脸位移」的一个平滑版本。**目前没有任何外部锚点说明 μ 和真实心理状态有关系。**

这不致命 —— programmatic weak supervision（Snorkel 那一套）本来就这么做。但弱监督范式的合法性来自两点：多个噪声源要**相互独立**，且至少要有**一次外部验证**。你现在两条都缺（三路打分器其实共享 q_* 质量项，不独立）。

→ 最省事的补法：找一个带标签的公开集（DAIC-WOZ / CMDC）跑一次 MPSE，报个相关系数就行。不需要好看，需要存在。

**A2. MPSE 的输入根本不是多模态**
`X = 文本 embedding(mean-pool) ⊕ [q_text, q_audio, q_video, microexpr_rate]`。音视频只贡献 3 个质量标量 + 1 个微表情率，**网络从没见过音频/视频的内容表征**。

由此带来两个问题：
- α 号称"三模态证据权重"，但网络只能从"文本 emb + 4 标量"里猜该信哪个模态，这个语义解释站不住。
- 更糟的是**逻辑倒置**：`y_soft` 是按 q_* 加权融合出来的，而 X 里也包含 q_*。网络完全可以从 q_* 反推融合权重，学一条「重建融合公式」的捷径，而不是学状态。这大概率就是 α 和质量标量高度相关的原因。

**A3. σ 是 aleatoric，不是 epistemic**
高斯 NLL 学出来的 σ 描述的是「给定 X，弱标签 Y 的离散程度」，即**数据噪声**。ASR 为空时 σ 飙高，是因为那类样本的弱标签本来就散，不是"模型不知道"。

用它做样本加权完全合理（降权噪声样本，这是对的）；但**不能讲成"模型的不确定性"**，一问就穿。诚实的说法：*heteroscedastic noise estimate，用于噪声感知的样本加权*。

**A4. Δμ 的因果解释开得太大**
"Δμ<0 → 上一条咨询师回复有效"。但 μ 只从当前 turn 的内容算，相邻两 turn 状态下降可能来自：话题转移、来访者自己想通、访谈的自然节奏、ASR 质量波动。归因给咨询师那一句需要控制变量。

→ 能站住的版本：**Δμ 是"回复后状态变化"的一个 proxy 信号，用来给样本排序 / 加权**，不是因果的有效性判定。改口径即可，不用改代码。

**A5. MPSE 完全没有时序 —— 最该改的一处**
整个命题是关于「多轮状态趋势」的，但 MPSE 是逐 turn 独立的 2 层 MLP。README 里自己写了 `z_t = f(h_t)` 的时序隐变量，代码里根本没有。

→ 在 `mpse/model.py` 的 fc 后加一个 GRU（按 session 组 batch），**改动最小、收益最大**。它把"逐句打分器"变成"状态追踪器"，Δμ 才真正有意义。这是我最推荐的一处架构更新。

### B 段（多模态前缀 SFT 生成器）

**A6. A→B 的连接是"软"的，而且重复注入**
μ/σ/α 走了两条路进 LLM：(1) `format_state_block` 把 μ/σ/α/p_ok 的 **Python 字典字面量**塞进 user prompt 文本；(2) α 缩放音视频软 token。

前者很脆 —— 模型在读 `{'dep': 0.2341, 'sad': 0.1897, ...}` 这种字符串，数值精度全是噪声。而且 prompt 里已经有 μ 了，`aux_mu_dim` 还让 projector 再去预测 μ，信息冗余。

→ 建议：μ 离散化成 3-5 档的离散 token（`[DEP:mid]`），或者干脆只留软 token 这一条路。

**A7. α 的维度对不上**
MPSE 输出 3 维 α (T/A/V)，`cache_builder` 只取 (A, V) 两维缩放前缀，**α_T 被直接丢掉**。而且 α 是 softmax 后的值（三者和为 1，各约 0.33），拿它直接乘软 token，效果更接近「把音视频前缀统一缩到 1/3」的常数衰减，而不是门控。

→ 门控应该用 sigmoid（各自独立 0~1），softmax 的语义是"证据分配"，不适合当增益。

**A8. 软 token 前缀的信息量太低**
一整个 turn 的音频 pooled 成 1 个向量、8 帧视频的 CLS 平均成 1 个向量，再各自线性展开成 8 个 token。**8 个 token 是从同一个向量线性映射出来的，本质上仍只有 1 个向量的信息量。**

跟 Q-Former / cross-attention 那类做法差距很大。你复现过 LQ-Former，正好知道差在哪：应该让 K 个 query token 去 cross-attend 音视频的**序列**表征（`encoders.py` 里 `return_sequence=True` 的路径已经留好了，只是没用）。

**A9. 没有 baseline，训练配置也太单薄**
冻结基座 + LoRA + 前缀，1 个 epoch，batch=1，没有 warmup / scheduler / 梯度累积，**没有纯文本 LoRA 对照**。所以现在无法说明前缀起了任何作用。

### 距离「一个完整的微调项目」还差多远

判断：**不远，而且差的全是同一类东西 —— 让它从"一条能跑通的 pipeline"变成"一个有结论的实验"。**

8 阶段闭环能跑通本身是硬本事，很多人做不到。但一个项目要讲得完整，需要三件套：**方法 + 能跑的系统 + 有对照的结果**。你有前两件，第三件是零。

必须补的清单（按必要性排，做完就算完整）：

| # | 事项 | 说明 |
|---|---|---|
| 1 | **按 session 划 train/val/test** | 绝对不能按 turn 划，同一段访谈会泄漏 |
| 2 | **纯文本 LoRA baseline** | 关掉前缀跑一遍，这是所有对比的地基 |
| 3 | **σ 校准图** | 样本按 σ 分箱，看每箱实际误差是否单调上升。**这一张图就能证明 σ 不是摆设** |
| 4 | **MPSE held-out 指标** | NLL / MAE / α 分布图 |
| 5 | **B 段 held-out perplexity + 定性样例** | 多模态 vs 纯文本各挑几条 |
| 6 | **四个消融开关** | 去音频 / 去视频 / 去 α 门控 / 去 σ 加权 |
| 7 | **README 写清结果** | 包括"什么没做到"，这比吹更值钱 |

**不需要**：真实临床数据、SOTA、论文、更大的模型。

### 架构更新建议（按投入产出比）

**高（建议做）**
- MPSE 加时序（GRU），把逐句打分器变成状态追踪器 [A5]
- σ 校准图 + 纯文本 baseline + 按 session 划分 [验证三件套]
- α 从 softmax 改成 sigmoid 门控，且三维全用上 [A7]
- 口径修正：σ 说成 noise estimate，Δμ 说成 proxy signal（零代码成本） [A3][A4]

**中（有时间就做）**
- 把音视频 embedding 真正接进 MPSE，让 α 名副其实 [A2]
- state block 从字典字面量改成离散 token [A6]
- 拿一个带标签公开集做一次外部验证 [A1]

**低（加分项）**
- Q-Former 式 cross-attention 替换 pooled 前缀（`return_sequence=True` 的路已经留好了） [A8]

---

## 6.8 数据集核实结果（2026-07-22）

> 起因：用户指出「脚手架数据无关」的说法站不住 —— σ 校准图要算「σ 分箱 vs 实际误差」，
> 那个"实际"目前只能拿 y_soft 当真值，而 y_soft 正是 MPSE 的训练目标，等于用循环论证验证循环论证。
> **正确顺序是：问题 → 数据 → 评估 → 架构。**

### 三个互相纠缠的命题（只能验证一个）

| | 命题 | 需要的真值 | 候选数据 |
|---|---|---|---|
| A | 无需人工标注也能做多模态状态估计 | 量表分 | DAIC-WOZ (PHQ-8) |
| **B** | **Δμ 能识别有效的咨询回复** | 咨询师行为 + 来访者状态方向 | **AnnoMI** |
| C | σ 加权改善下游微调 | **不需要外部标签，自足** | 任何对话数据 |

### AnnoMI（已核实）
- **133 段**专家标注的动机式访谈对话；其中 110 段 high-quality，含 8,800+ utterance
- 字段（`AnnoMI-simple.csv`）：`transcript_id`, `mi_quality`, `video_title`, `video_url`, `topic`,
  `utterance_id`, `interlocutor`, `timestamp`, `utterance_text`, `main_therapist_behaviour`, `client_talk_type`
- `AnnoMI-full.csv` 另有：`annotator_id`, `therapist_input_exists/subtype`, `reflection_exists/subtype`, `question_exists/subtype`
- `client_talk_type` ∈ {change, neutral, sustain} —— **这就是 Δμ 方向的现成真值**
- 仓库：https://github.com/uccollab/AnnoMI

**为什么它跟本项目对得离谱**（不只是有真值）：

| 现有 pipeline 阶段 | AnnoMI 直接替代 |
|---|---|
| 能量 VAD 切 turn（最脆的一环） | `timestamp` 直接给 turn 边界 |
| speaker diarization（`diarize.py` 没接线） | `interlocutor` 直接给角色 |
| faster-whisper ASR | `utterance_text` 是专业转录稿（ASR 转为可选，或用来做质量校验） |
| **人工补 `therapist_reply`（最痛的一环）** | **咨询师的真实回复本来就在数据里，每一条都有** |
| 无真值 | `client_talk_type` + `mi_quality` 双层真值 |

→ **5.5 节里"接 pyannote"那条决策，选 AnnoMI 之后直接作废**，因为切分和角色都是白送的。
→ 6.5 A1 的循环论证窟窿，靠 `client_talk_type` 补上。

**风险（必须先验证）**
1. ⚠️ **不含视频文件**，只给 `video_url`（YouTube 链接）+ timestamp。要自己下载。数据集 2022 年发布，**部分链接大概率已失效，开工前必须先跑一遍链接存活率**。
2. 下载 YouTube 视频有 ToS 灰色地带，学术用途常见但要心里有数。
3. AnnoMI 是**演示性质**的访谈（专家为教学录制），不是真实临床会话。**跟用户自己那批"演的"数据性质相同**，区别在于：演的人是专家、标注是专家做的、而且是公开基准，别人也用它发论文。所以合法性来自"公共基准"而非"真实临床"。
4. 许可条款未从仓库页确认，只知道描述为 "publicly and freely available"。**要在下载前确认。**
5. 只有 23 段 low-quality，做 high/low 对比时这一侧样本偏少。

### DAIC-WOZ（已核实）—— ⚠️ 会废掉现有视频链路
- 需签 EULA 申请，session ID 300–492
- 提供 16kHz wav、转录稿、PHQ-8 分数（0–24，含二分类标签）
- ⚠️ **不提供原始视频**。出于隐私，只给 OpenFace 预抽的特征：68 个 2D/3D 面部关键点、HOG、面部动作单元 AU、4 个 gaze 向量、head pose
- **后果**：`features/video_features.py`（MediaPipe FaceLandmarker 吃 mp4）和 `mm/encoders.py` 的 `CLIPVideoEncoder`（吃视频帧）**两个都得推倒重写，CLIP 直接出局**。选 A 等于放弃现有视频链路。
- 另外 PHQ-8 是 **session 级**标签，而 μ 是 turn 级，中间那层聚合怎么做本身就是可争议的设计。

### CMDC
中文数据集，与 5.5 节「走英文」的既定决策冲突，本轮不考虑。

---

### ✅ AnnoMI 可行性探针结果（2026-07-22 实测，非二手资料）

直接拉了 `AnnoMI-simple.csv`（2.4MB）和 `AnnoMI-full.csv`（3.8MB）跑的统计。

**规模**
- 9,699 utterance / 133 段对话（high 110，low 23）
- therapist 4,882 条，**client 4,817 条**（MPSE 只吃 client turn）
- 每段 turn 数：min 6 / median 47 / max 598 / mean 72.9

**标注分布**
- `client_talk_type`（仅 client turn）：neutral 3,102 (64.4%)、change 1,174 (24.4%)、sustain 541 (11.2%)
- `main_therapist_behaviour`（仅 therapist turn）：other 1,586、question 1,386、reflection 1,296、therapist_input 614

**★ 最关键的一个数：high / low 质量的对比是真实存在的**

| mi_quality | n | change | neutral | sustain | **change:sustain** |
|---|---|---|---|---|---|
| high | 4,398 | 25.1% | 64.1% | 10.9% | **2.30** |
| low | 419 | 17.2% | 67.8% | 15.0% | **1.15** |

高质量会话的 change:sustain 比是低质量的 **2 倍**。这意味着「好的咨询回复 → 来访者状态朝好的方向走」这个信号**在数据里真实存在且可测**。
→ 命题 B 有得做。如果这个比值是 1:1，B 就该当场放弃。

**视频可获取性（实测 oEmbed）**
- 119 个唯一视频 URL：youtube.com 112、youtu.be 4、vimeo.com 3
- **117 / 119 存活 = 98.3%**（3 个 vimeo 全部存活，之前 404 是我用 YouTube oEmbed 测 vimeo 的误判）
- 失效仅 2 个 YouTube 链接（1 个 404、1 个 403），影响 **2 / 133 段对话、66 / 9,699 条 utterance**
- 结论：**远超"六成"的止损线，视频链路可以保住，CLIP / MediaPipe 都不用动**

**⚠️ 唯一的真实工程约束：时间戳粒度**
- `timestamp` 是 `HH:MM:SS`，**秒级精度，且只标起点没有终点**。turn 的时长只能由「下一条的起点」推出。
- 实测 client turn 的时长分布：**0 秒 13.5%（640 条，切不出音视频片段）**、1 秒 23.0%、>=2 秒 63.6%
- 另有 10 处时间戳倒序（负时长），需要清洗
- → **可用的 client turn 约 3,026 条**（>=2 秒那部分）。即便如此，仍是现有 371 条的 **8 倍**，完全够用。
- → 备选方案：对短 turn 做窗口外扩（会跨到对方说话），或在窗口内跑一次 VAD/强制对齐拿精确边界。**建议第一版直接按 >=2 秒过滤，别做花活。**

**许可**
- 仓库**没有 LICENSE 文件**。README 只要求引用 ICASSP 2022 那篇论文（Wu et al., *Anno-MI: A Dataset of Expert-Annotated Counselling Dialogues*）。
- 即：事实上自由使用 + 需引用，但**没有正式的开源许可条款**。视频本身的版权归各 YouTube/Vimeo 上传者，跟数据集分开。

**历史注记**：用户当初就想选 AnnoMI，被原单位老板以「内容太多而且是英文」否掉。现在项目脱离原单位，这条否决不再成立。

---

### ✅ Premise 实测（2026-07-22，纯 gold label，零模型）——决定了架构

在搭任何模型前，先在专家标签里验证命题的前提。四个发现，直接定架构：

- **F1 逐句归因是空的**：相邻 client turn 有 65–73% 状态不变；reflection（MI 里最标准的"好动作"）对下一句状态的净效应 = **−0.002**（reflection vs other：P(change) 0.232 vs 0.233，z=−0.10）。→ **"好回复 → 下一句变好"在数据里不存在。**
- **F2 会话弧线真实存在**：改变倾向 change/(change+sustain) 从早段 ~50% 升到晚段 ~82%。→ 状态动态在**会话尺度**，不在逐句。
- **F3 判别信号是形状不是水平**：net 改善不分高低（都爬到 ~80%）；**弧线形状**分——high 单调爬升 53→68→84%，low 中段塌陷 48→**26**→80%。
- **F4 咨询师动作预测力≈0**：reflection 密度 vs change:sustain 的 Spearman = **0.072**。
- 会话级 change:sustain：high 2.00 vs low 0.33（强，这条是全样本，稳）。
- ⚠️ 保留：low 仅 23 段，"中段塌陷"这个形状细节是提示性、非定论。

### ✅ 架构锁定（2026-07-22）—— 数据 → 决定

| 数据发现 | 架构决定 |
|---|---|
| F1 | **砍掉**转移头 + 咨询师动作条件（v1-强原本的核心，靶子是空的）|
| F2 | **保留时序模型（GRU over turns）**：职责是出会话轨迹，不是逐句归因 |
| F3 | 读出层 + H2 瞄准**轨迹形状特征**（单调性/中段回撤/平滑度），非净漂移 |
| F4 | 咨询师动作**降级**为事后协变量，不进模型输入 |
| F5（neutral 占 64%，逐句噪声大）| 序列模型的另一理由：去噪 |

**最终架构（比 v1-强更简单，数据逼出来的）**：
`MPSE = 序列估计器(GRU) → 逐句 μ_chg + σ → 拼成会话轨迹 → 读轨迹形状`。
无转移头、无动作条件。MPSE 直接吃 Whisper/CLIP 序列表征（缺口一），编码器与生成器共享。

**锁定命题（3 条，全有 gold label 兜底）**：
- H1 估计器有效性：μ_chg 逐句还原 talk_type
- H2 轨迹形状：形状特征区分 MI 质量
- H3 σ 双重角色：非循环校准（按 σ 分箱看 H1 的 AUC）+ 低质量会话 σ 更高

**定位**：项目导向，非论文导向。评估器是主角（笔墨集中在验证），生成器是 demo（证明信号能驱动生成，不跟 SOTA 比）。

---

## 6.9 评估与架构方案（已锁定）

完整文档：**`docs/eval-design.md`（v2，已锁定）**。上面 premise 实测 + 架构锁定两节是它的浓缩。
命题、架构、维度、划分、假设 H1/H2/H3、实施顺序都在里面，开工按那份走。

---

## 7. 路线图

**顺序的理由**：数据定架构，架构定评估。所以先钉数据(✅已做)、锁架构(✅已做)，再按 `docs/eval-design.md` §5 落地。

### 已完成
- [x] 代码统一到 `C:\Users\qmn20\Yuan-chat`，拍平嵌套目录，加 `.gitignore`（分支 `sync-latest-code`，**尚未 push**）
- [x] 方法架构评估（§6.5，9 条）
- [x] 笔记归拢：旧存档并入本文件并删除原文件
- [x] 数据钉死：AnnoMI（§6.8 探针，98.3% 视频存活）
- [x] premise 实测（§6.8）+ 架构锁定：序列估计器 + 轨迹形状 + σ 双重角色
- [x] 评估与架构方案 v2 落盘（`docs/eval-design.md`）

### 服务器现状（AutoDL，2026-07-22）
- 实例：`autodl-mpse`（SSH 别名已配，VSCode 可直连，密钥免密码）。西北B区,vGPU-32GB(物理 4080S 16G + 超分,CUDA 见 32GB),数据盘 250G。
- 镜像 PyTorch 2.1.2+cu121 / py3.10 / CUDA 可用。`scripts/setup_server.sh` 装通,**`smoke_test.sh` 全绿**(torch/GPU/Whisper/CLIP/MediaPipe/librosa/项目导入)。编码器已缓存在 `/root/autodl-tmp/hf`(1.5G)。
- **网络约束(国内)**:
  - HF 模型:直连被墙 → `HF_ENDPOINT=hf-mirror.com` + `HF_HUB_DISABLE_XET` 或 `source /etc/network_turbo`(学术代理)可下。已验证。
  - YouTube 视频:服务器学术代理走不通(MITM 自签证书 + 503 限流)。**改在用户本机下载**(用户人在国外,直连 YouTube)→ scp 上传服务器。
- **视频下载方案(已跑通)**:用户本机 `python scripts/download_annomi_videos.py --skip_wav --format "best[height<=480]/best"`(渐进式单文件,~360p,不需要 ffmpeg,~13M/个、131 个共 ~1.5G),下完 scp 到服务器,服务器用 ffmpeg 抽 wav。脚本已加 `--skip_wav`/`--format`/`--insecure` 开关、yt-dlp 用 `python -m yt_dlp` 调用(免 PATH 依赖)。
- ⚠️ 用户明确:**评估器就是为多模态设计的,视频不能砍**。所以不走"纯文本版",直接奔多模态。文本 H1 只作为 baseline 顺带产出。

### 弱标签定夺
- **chg 弱标签**:词典(MI DARN-C 词表)实测太弱——76% turn 得分 0、AUC 仅 0.58(<0.6 可用线)。→ **改用模型**(zero-shot NLI，如 bart-large-mnli，"想改变 vs 不想"打分)。避免了用死标签训出死模型。
- aro:音频韵律(现成)。val/eng:面部(MediaPipe),待视频到位后跑维度探针定去留。
- 注意:gold talk_type 只作评估,**绝不进训练**(否则泄漏)。
- chg zero-shot(bart-large-mnli)实测:**AUC 0.667**(词典 0.58),可用。写在 `data/annomi/turns_chg.jsonl`。

### ★ 成功判据的重新定义(2026-07-22)
弱标签是**文本**的(AUC 0.667)。MPSE 的价值不是复制它,是用**多模态 + GRU 时序去噪**。
→ **核心判据:μ_chg(多模态+时序,在 0.667 噪声弱标签上训)对 gold 的 AUC 能否 > 0.667。**
能 = 模型加了信息(多模态融合 + 时序平滑),非循环、且证成 GRU+多模态架构的必要性(呼应 F5)。
配套对照:文本-only MPSE 的 μ vs gold,看多模态相对纯文本涨多少。
⚠️ 潜在诚实结果:chg 高度语义化(文本主导),多模态对 chg 可能提升有限——那也是如实报告的发现,不强凑。

### ★ 第一次端到端结果 + 战略性发现（2026-07-22）
**里程碑**:AnnoMI → 三模态特征(text MiniLM384 / audio Whisper768 / video CLIP768) → MPSE(GRU+sigmoidα+异方差) → H1/H2/H3,**整条端到端跑通**,128 session/2933 turn。

**结果(chg,袋外 CV,AUC=区分 change vs sustain)**:
| | AUC vs gold | μ std |
|---|---|---|
| bart 弱标签(训练目标) | 0.667 | 0.19 |
| MPSE 多模态 | **0.592** | 0.12 |
| MPSE 纯文本 | 0.590 | 0.12 |
| MiniLM→gold 监督线性探针(上界) | **0.608** | — |

**两个确认的发现**:
1. **MiniLM 文本编码器是瓶颈**:监督上界仅 0.608 < 弱标签 0.667。MPSE 0.59 贴着这个上界,不是欠拟合。GRU/无GRU、NLL/MSE 四种消融全是 ~0.59 → 与架构无关。→ **需换强文本编码器**(bart 自身 embedding / LLM),chg 才能逼近 0.667。
2. **多模态对 chg 没用**(0.592≈0.590):chg 是文本语义构念,音视频不携带"改变意愿"语义。

**★ 战略性张力(核心)**:唯一有 gold 验证的维度(chg)是**文本主导**,多模态帮不上;而多模态能帮的维度(aro 韵律 / val 面部)在 AnnoMI **没有 gold**。→ "多模态在 chg 上胜过文本"这个主张,数据说它**不成立**。
**破局候选**:
- A. 换强文本编码器让 chg≈0.667,主线改讲 **σ 校准(H3)+ 轨迹(H2)**,如实报告"chg 上多模态无增益"。
- B. 多模态价值移到**生成器**(音视频语境→更好回复,perplexity)。
- C.(有意思)测**多模态状态(chg+aro+val)的轨迹**能否比 chg-only 更好地区分 mi_quality(gold)。→ 多模态用"质量区分"证明价值,绕开"aro/val 无 gold"。gold=mi_quality 是现成的。

### 本机环境（重要）
`C:\Users\qmn20` 是裸 Windows：**只有 numpy，无 torch/transformers/ffmpeg/yt-dlp**。
→ 所有 torch/模型/编码/训练代码都得在**租的 H800 服务器**上跑，本机只能写+跑纯 numpy/csv 逻辑。
→ 因此开发策略：本机先写能自测的纯 numpy 部分（数据适配、划分、评估脚本），torch 部分写好待服务器验证。

### 进度（按 `docs/eval-design.md` §5）
- [x] **1. `data/annomi.py`**：CSV→turns，清洗。**产出 3026 可用 client turn**（=旧数据 8 倍），109 段 ≥8-turn（H2 样本，high 93/low 16），131/133 视频存活。
- [x] chg 维度：premise 已确认活着，**锁定为验证维度**（不必再写 probe）
- [x] **4. `eval/split.py`**：按 session 分层 5 折 + holdout → `splits.json`，校验无泄漏
- [x] **7. `eval/{metrics,h1,h3}.py`**：纯 numpy，**已用合成数据自测通过**（H1 判定逻辑、H3 非循环校准曲线、置换检验、bootstrap CI 都对）。schema 已钉死：predictions jsonl = `{session_id,turn_id,mi_quality,talk_type,mu:{chg},sigma:{chg}}`
- [ ] 2b. `probe_dims`（val/aro/eng 部分）：**阻塞于音视频**，等下载后做
- [ ] 3. 下载 117 视频抽 wav ← **关键路径,需服务器 + yt-dlp/ffmpeg**
- [ ] 5. 编码器(Whisper/CLIP 序列) + MPSE(GRU + sigmoid α 门控 + 真多模态输入) 改造（torch，服务器验证）
- [ ] 6. 训 MPSE + upgrade（服务器）
- [ ] 7b. h1/h3 跑真实预测（分水岭）+ `eval/h2.py` 轨迹形状
- [ ] 9. 生成器 demo + perplexity；10. 重写 README

**当前关键路径 = 服务器 + 视频下载**（步骤 3）。它一通，后面 5/6/7b 才能跑。本机能写的纯 numpy 件已基本铺完。

### 顺带要修的工程债（穿插进上面，非独立阶段）
- [ ] `paths.models_root` 收编硬编码模型路径 + `face_landmarker.task`
- [ ] `merge_mm_index.py` 改相对路径
- [ ] `build_sft.py` 补空 reply 过滤开关
- [ ] 口径修正：σ→noise estimate、Δμ→proxy/轨迹信号（改 README/docstring）

### 挂起
- [ ] 生成器基模选型：迭代期中等模型、展示期大模型；待架构就绪后联网拉当前开源清单比对
- [ ] 仓库转公开 + push（用户已定：**做完再转**；目前私有）
- [ ] （可选）用 `AnnoMI-full` 细行为子类再挖一次逐句效应 [premise B 分支，预判弱信号]

---

## 8. 关联信息

### 简历（GSOE9011 A03，已于 2026-07-10 交）
- 成品：`C:\Users\qmn20\Desktop\9011CV\Mengnan Qin - GSOE9011 A03 Submission.pdf`（英文 CV 2 页 + 岗位分析表 1 页）
- 目标岗：TikTok — ML Engineer Graduate (Trust & Safety), 2027 Start, Sydney
- 简历里 MPSE 那段有个**编的数字"多模态融合使多轮状态估计一致性提升约 11%"**，以及"~20 段访谈、300+ 轮对话"。
  - "~20 段 / 300+ 轮"→ 实际 19 段 / 371 轮，**是真的**。
  - "11%" 是编的。当时判断是"作业不投递所以可接受"。**如果这份简历要拿去真投，必须先做完第 7 节第 3 条（补 eval）拿真数字替换。**
- LQ-Former 是用户真实独立复现过的，但**没用进 Yuan-chat**，不能挂在 MPSE 这个项目下。

### 职业策略结论（来自 6-11 那次深聊）
- 甜区 = **多模态 + agent，应用/工程层算法**。避开"微调当卖点"（半商品化）和"用 RL 训 agent"（前沿红海）。真正高杠杆但没那么挤：数据、eval、推理优化。
- 端到端自动驾驶：这一届硬冲正门核心算法岗基本没有；**侧门**（进 AD 公司做大模型/多模态/数据岗 → 1-2 年内转）是真实进法。把端到端从"这个月就要"改成"两三年的方向"。
- 秋招约 2026 年 8 月开始 —— **就是下个月**。Yuan-chat 是他手里最硬的那张牌，"修出 eval、讲成多模态大模型的故事"是当时定下的前置条件。

### 相关文件位置
- 旧存档笔记：`C:\Users\qmn20\claude-yuanchat-notes.md`
- HiHappy 论文（ACL 在投，用户是共同作者）：`Desktop\Yuan-chat\待打包材料\MLLM-mental_health_care\HiHappy_*.pdf`
- 文献库：`Desktop\Yuan-chat\待打包材料\MLLM-mental_health_care\`（Q-former、AffectGPT、scoping review 等）、`E2E papers\`（UniAD、VAD、OpenDriveVLA 等）
- 咨询转录稿（人工填 therapist_reply 的来源）：`Desktop\Yuan-chat\待打包材料\Yuan-chat\可用对话\*.pdf`
