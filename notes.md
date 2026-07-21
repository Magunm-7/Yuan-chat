# Yuan-chat / MPSE · 项目进度笔记

> 最后更新：2026-07-22
> 用途：新开对话时读这份，能恢复到"已经把整个项目和代码吃透"的状态。
> 上游材料：`C:\Users\qmn20\claude-yuanchat-notes.md`（2026-06-11 的旧存档，部分结论已过期，见第 5 节）

---

## 0. 上下文来源说明

原始那段深聊（对话标题 "Yuan-chat"，2026-06-11）**已经被 Claude Code 自动清理掉了**（transcript 默认只留 30 天）。
本笔记重建自三个还活着的来源：

1. `C:\Users\qmn20\claude-yuanchat-notes.md` —— 当时手动落盘的存档笔记（唯一幸存的浓缩版）
2. 会话 `2d026650-f43b-4c4f-adf6-886e3457d47b`（2026-07-09 11:04–13:10）—— 那次为了改简历，重新读了一遍旧对话 + 通读了 git 仓库全部代码
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

## 6. 当前已知问题（按优先级）

### P0 —— 直接污染训练结果
1. **102/371 个训练样本的 target 是字面量 `(FILL_THERAPIST_REPLY_HERE)`**
   `build_sft.py` 在 `therapist_reply` 为空时填这个占位符，而 `run_full_loop` 没过滤，`merge_mm_index` 也照单全收。**模型有 27.5% 的样本在学着输出这个占位符。**
   → 修法二选一：把 103 条缺的回复补齐；或在 `build_sft` / `merge_mm_index` 里直接跳过空 reply 的样本（`build_sft_from_pairs` 里已有 `skip_if_no_pair` 逻辑，抄过来即可）。
   → 特别注意 **S0001 整段 18 个 turn 一条 reply 都没填**。

2. **完全没有 eval**。没有 hold-out、没有任何指标、没有消融。项目最缺的一块，也是"讲成一个能拿出手的项目"的最后一道坎。

### P1 —— 工程健康度
3. `mm_cache/ALL/mm_index.jsonl` 里的 `npz_path` 是 **Linux 绝对路径** `/home/qmn/Yuan-chat/MPSE_FULL_LOOP_PROJECT/...`，在 Windows 上一条都读不到。要么改成相对路径，要么加运行时 rebase。
4. **绝对路径写死**遍地：`configs/default.yaml` 里 6 处模型路径 + `features/video_features.py` 里 `face_landmarker.task` 的路径。
5. `mm_cache/S0003` 有 48 个 npz 但 index 只有 12 条（旧运行的残留没清）；`mm_cache/S0009` npz=23 但 index=24（**少一个 npz，会在读取时抛异常**）。
6. **代码没推 GitHub**：仓库停在 2026-02-03，README 却描述着 2 月底的新代码。仓库现在是"文档超前于代码"的状态，别人（和面试官）clone 下来跑不通 README 写的流程。
7. 目录 cruft：嵌套的 `mpse_full_loop_project/`、带空格的 `data & output/`。

### P2 —— 架构与描述不符（要诚实）
8. **MPSE 这一级"多模态"名不副实**：X 只有「文本 embedding + 4 个质量标量」，真正的音视频 embedding 只在 B 段（前缀 SFT）出现。所以 α 是"三模态证据权重"这个说法在 MPSE 这级站不太住。
   → 要么把 video/audio embedding 真正接进 MPSE（存档笔记里的候选 ⑤），要么在描述上改口径。
9. `diarize.py` 写完没接线；dialogue 模式用的是"VAD 段严格交替分配 role"这种脆弱假设，而且 `dialogue.enabled=false`，实际走的是"人工剪掉机器声音 + 人工补 therapist_reply"的半自动流程。
10. `microexpr_rate` 在旧数据里饱和成常数 1.0（`clip(m/0.015, 0, 1)` 阈值定太低），新数据要复核。

---

## 7. 下一步候选（按启动成本排，别一次全做）

1. **[10 分钟]** 决定代码的唯一真相源：把桌面那份新代码搬进 git 仓库、拍平嵌套目录、提交推送。现在这个"两份代码"的状态是所有混乱的根源。
2. **[30 分钟]** 修 P0-1：过滤掉 102 条占位符样本（或补齐），重新 merge，重训一版。这是**投入产出比最高的一件事**——直接改善模型输出质量。
3. **[半天]** 补 eval：MPSE 切 hold-out 出几个数字（NLL / 校准曲线 / σ vs 误差相关性）+ α 分布图 + Δμ 轨迹图。**有了这个，项目才有"结果"可讲。**
4. **[半天]** 绝对路径全部收进 config，加个 `paths.models_root`；`mm_index` 改相对路径。让项目在别人机器上能跑。
5. **[1 天]** 把 video/audio embedding 真正接进 MPSE，让 α 名副其实；顺便做一个"纯文本 baseline vs 多模态"的对比——这正好是 eval 的天然消融实验。
6. **[持续]** 把项目讲成"多模态弱监督 + 不确定性感知"的简历/面试故事（见第 8 节）。

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
