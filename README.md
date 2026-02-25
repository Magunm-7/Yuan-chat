# MPSE
- 底层原理：
假设有三轮对话，人：S1，S2，S3，模型：Y1，Y2。那三段对话的顺序就是S1，Y1，S2，Y2，S3.我们设计评估器的目的就是为了挑选出好的Yi使得从Si到Si+1的状态之中，某些指标z有了下降，比如抑郁指标dep，我们希望在多轮对话的情况下dep处于一个下降的趋势，那我们就要量化到底哪些回复Y达到了期望目标
- 输出参数：
评估器会对人说的话赋予额外三个参数：μ，σ，α

# 多模态证据融合与状态估计

> 目标：对每个 *human turn* 片段，从文本/语音/视频三种模态提取特征，并进行门控融合，输出可解释的状态估计（均值/不确定性/模态权重）。

---

## 1. 特征提取

对一个 human turn 片段，针对三个模态提取特征：

- **文字**：内容编码 `encode`  
  $$
  \text{encode}(\text{text}) \rightarrow h_t^{\text{text}}
  $$

- **语音**：时间段 \([t_0, t_1]\)（可加 \(\pm \delta\) 扩展）编码  
  \[
  \text{encode}(\text{audio}_{[t_0,\,t_1]}) \rightarrow h_t^{\text{audio}}
  \]

- **视频**：时间段 \([t_0-\delta,\, t_1+\delta]\) 编码  
  \[
  \text{encode}(\text{video}_{[t_0-\delta,\,t_1+\delta]}) \rightarrow h_t^{\text{video}}
  \]

---

## 2. 初始化证据权重（模态门控融合）

定义模态证据权重（门控系数）：

\[
\alpha_t = (\alpha^{\text{text}},\;\alpha^{\text{aud}},\;\alpha^{\text{vid}})
\]

融合得到统一表征：

\[
h_t
= \alpha_t^{\text{text}}\, h_t^{\text{text}}
+ \alpha^{\text{aud}}\, h_t^{\text{aud}}
+ \alpha^{\text{vid}}\, h_t^{\text{vid}}
\]

解释性：\(h_t\) 是统一表征，\(\alpha_t\) 用于解释该步主要由哪个模态提供证据（哪种证据权重更大）。

---

## 3. 时序隐变量与异方差回归输出

将融合表征映射到时序隐变量：

\[
z_t = f(h_t)
\quad\text{（把输出变为隐变量）}
\]

异方差回归输出（均值与方差）：

\[
\mu_t = W_{\mu} z_t,\qquad
\sigma_t^2 = \mathrm{softplus}(W_{\sigma} z_t)
\]

一次前向输出包含：

- \(\mu_t\)：状态估计均值（可用于趋势/变化判断）
- \(\sigma_t^2\)：不确定性（方差）
- \(\alpha_t\)：模态证据权重（可解释性）

---

## 4. 用 \(\mu_t\) 做对话“跨度/变化”判断（以 dep 为例）

对连续 turn 的变化幅度：

\[
\Delta \mu_{t\to t+1} = \mu_{t+1} - \mu_t
\]

以 \(\mu^{\text{dep}}\)（例如 depression 维度）为例，可用阈值 \(\zeta\) 判断：

- 若  
  \[
  \Delta \mu^{\text{dep}} < -\zeta
  \]
  则认为出现 **有效回复**（朝期望方向变化）

- 若  
  \[
  \Delta \mu^{\text{dep}} > \zeta
  \]
  则认为出现 **无效回复 / 反向变化**（原笔记处字迹略糊，此处按语义整理）

---

## 5. 用 \(\sigma_t\) 表达不确定性

\[
\sigma_t^2 \text{（或 } \sigma_t \text{）越大} \Rightarrow \text{模型对该步估计越不确定}
\]

直观解释（笔记语义）：

- 如果 \(\sigma_t\) 很大，说明估计器对新信息不确定；
- 可能意味着该段对话存在异常，或对应不同的心理/对话模式。

---

## 6. 用 \(\alpha_t\) 做模态依赖解释与数据平衡

\(\alpha_t\) 可以解释该步主要依赖哪种模态证据，例如：

- 若 \(\alpha^{\text{vid}}\) 很高，但该段视频处于人脸模糊/遮挡期，则可能是可疑样本（高视频权重但视频质量差）。

也可用于数据策略与训练平衡：

- 利用 \(\alpha\) 识别并采样更多 **audio-dominant** 或 **video-dominant** 样本，
- 从而促进三模态在训练中的贡献更均衡，避免模型偏向单一模态。

---
	​
# config/default.yaml
- 顶层session：
```yaml
session:
  id: "S0001"
```
默认跑哪一个session，现在的训练逻辑是每一个样本是一个单位，先对单个样本进行数据处理，然后进行汇总再去训练

- pipeline
```yaml
pipeline:
  run_extract_wav: false
  run_build_turns: false
  run_build_mpse_npz: false
  run_upgrade: false
  run_train_mpse: false

  run_build_sft: false
  run_mm_cache: false

  run_train_mm_sft: true
  # 关键：保护你手改的 turns_upgraded.jsonl
  # overwrite: 每次都重建（会重置）
  # skip_if_exists: 如果 outputs/upgrade/<sid>/turns_upgraded.jsonl 已存在就不覆盖
  upgraded_policy: "skip_if_exists"
```
各个部分对应的开关
run_extract_wav: 把视频中的音频提取出来，因为后面切分对话主要靠的是音频
run_build_turns: 构建切分对话数据集
run_build_mpse_npz: 构建用于训练mpse的数据集
run_upgrade: 训练好MPSE之后用MPSE去升级数据集
run_train_mpse: 训练mpse
run_build_sft: 构建用于SFT的数据集
run_mm_cache: 把多模态特征算出来并缓存落盘
run_train_mm_sft: 以上所有准备完毕后进行微调

- 数据输入路径以及输出路径：
```yaml
paths:
  raw_video: "data/raw/{session_id}.mp4"
  work_dir: "data/derived/{session_id}"
  outputs_dir: "outputs"
```
原始视频文件放在raw下，derived文件包含的是视频对应的原始音频文件，以及切分后的对话文件

- segment: 分段与VAD参数
```yaml
segment:
  target_turns: 0
  wav_sr: 16000
  vad:
    frame_ms: 30
    thr: 0.00004
    min_speech_ms: 250
    min_silence_ms: 800
    merge_gap_sec: 0
    force_single_turn: false
```
上一段说到derived文件包含了切分后的对话文件，这里就是在设置切分的规则，需要提前知道的是我们现在处理的视频格式，是剔除了机器的回复，原因在于如果使用人机对话的原始视频，采取人一句机器一句的交替对话模式，仅靠现有的工具很难识别完整，所以我直接对原始（人和机器都有出声）视频进行裁剪，把机器说话的声音剔除，仅保留人说话的部分，再在后期把机器说话的部分在outputs/upgrade/.json文件里补上，实测效果是比直接处理人机对话要好
wav_sr: 16000：统一音频采样率（和 README 对齐）。
target_turns: 0：0通常代表“不限制/不裁剪”，靠 VAD 自然切分。
vad.frame_ms：VAD 分帧长度（30ms）。
vad.thr：能量阈值（越小越敏感，越容易把噪声当语音）。
min_speech_ms：最短有效语音段（过滤碎片）。
min_silence_ms：切分所需静音长度（越大 → 段更长、切得更少）。
merge_gap_sec：相邻段之间小间隔合并阈值（0 表示不合并）。
force_single_turn：强制整段作为一个 turn（debug 用）。
这里会直接决定 turns.jsonl 的质量；turns.jsonl 又是后面 MPSE/upgrade/SFT 的“地基”

- dialogue 人机交替对话模式
```yaml
dialogue:
  enabled: false
  mode: "alternating"
  start_role: "assistant"
  keep_role: "user"
  export_all_turns: true
  export_pairs: true
  target_user_turns: null
```
这里就是上一段说的直接对未剪辑的原始视频进行处理，效果不好，所以这里的开关直接关掉，不多做解释

- asr 语音转文字
```yaml
asr:
  enabled: true
  model_dir: "/home/qmn/.../models/faster_whisper/pengzhendong/faster-whisper-small"
  device: "cuda"
  compute_type: "int8"
```
这里就是在处理视频时候用到的语音转文字工具，这里的模型路径是写死的，到时候要改。精度也可以修改，int8是为了更省显存的做法

- agents：三路弱监督开关
```yaml
agents:
  use_llm_text_rater: true
  use_audio_heuristic: true
  use_video_heuristic: true
```

- llm： 基座模型
```yaml
llm:
  model_dir: "/home/qmn/.../models/Qwen3-8B/Qwen/Qwen3-8B"
  device: "cuda"
  max_new_tokens: 128
```
用的8B的，目前的训练数据量是偏小的，但是使用的是lora微调，所以也还说得过去

- 心里维度和指标+默认阈值
```yaml
indices:
  names: ["dep", "sad", "anx", "stress", "microexpr_rate"]
  tau:
    dep: 0.30
    sad: 0.30
    anx: 0.30
    stress: 0.30
    microexpr_rate: 0.30
```
设置了4+1个指标维度，分别是dep, sad, anx, stress，和microexpr_rate，后者是一个面部表情的抓取率，反应的是当前视频的质量问题

- mpse: 训练该评估器用到的超参
```yaml
mpse:
  epochs: 5
  batch_size: 8
  lr: 3.0e-4
  hidden_dim: 256
  dropout: 0.1
  use_pretrained_encoders: true
  encoders:
    text_model_dir: "/home/qmn/.../models/Qwen3-8B/..."
    audio_model_dir: "/home/qmn/.../models/whisper-small/..."
    video_model_dir: "/home/qmn/.../models/clip-vit-base-patch32/..."
```
这个评估器就是我们用来训练 (mu, sigma, alpha) 的核心模块
三模态的编码器路径也是写死的，需要更改
audio_model_dir 这里用的是 HF whisper-small（与 ASR 的 CT2 faster-whisper 不同体系）

- upgrade: 用训练好的MPSE模块对原始切分数据进行升级
```yaml
upgrade:
  sigma_lambda: 2.0
  sigma_max: 0.12
  inject_state_tokens: true
```
sigma_lambda：不确定性惩罚系数（sigma 越大 → 权重越低）。
sigma_max：只在 sigma 小于它时才“信任 improvement”。
inject_state_tokens: true：把状态（可能是 indices / mu / sigma / alpha）注入到 SFT prompt（这会影响最终训练数据格式）。

- sft: 文本 SFT baseline
```yaml
sft:
  enabled: false
  max_seq_len: 1024
```
可选 text-only baseline 的对应实现

- mm: 多模态特征提取配置
```yaml
mm:
  enabled: true
  whisper_dir: "/home/qmn/.../whisper-small"
  clip_dir: "/home/qmn/.../clip-vit-base-patch32"
  n_frames: 8
  k_audio: 8
  k_video: 8
  device: "cuda"
```
whisper_dir/clip_dir：HF 模型目录（绝对路径）。
n_frames：视频采样帧数。
k_audio/k_video：“压缩成多少个软 token”的长度。
这块会直接影响“mm_cache / mm_prefix.pt”等产物。

- teacher: 咨询师回复模型
```yaml
teacher:
  enabled: false
  base_model_dir: "/home/qmn/.../Qwen3-8B"
  device: "cuda"
  max_new_tokens: 128
```
这部分是一开始在构造最小闭环的时候加的，现在用不到了，保持关闭就行

- mm_sft: 多模态SFT训练
```yaml
mm_sft:
  enabled: true
  base_model_dir: "/home/qmn/.../Qwen3-8B"
  out_dir: "outputs/mm_sft/{session_id}"
  batch_size: 1
  lr: 0.0002
  epochs: 1
  max_len: 1024
```
模型路径是绝对路径需要更改，输出路径是相对路径不需要更改


# 需要修改绝对路径的地方
- config/default.yaml 内包含的路径有：
  - asr: model_dir 使用的模型是 pengzhendong/faster-whisper-small
  
  - llm: model_dir 使用的模型是 Qwen3-8B/Qwen/Qwen3-8B
  
  - mpse: text_model_dir 使用的模型是 Qwen3-8B/Qwen/Qwen3-8B
          audio_model_dir 使用的模型是 openai-mirror/whisper-small
          video_model_dir 使用的模型是 openai-mirror/clip-vit-base-patch32
  
  - mm: whisper_dir 使用的模型是 openai-mirror/whisper-small
        clip_dir 使用的模型是 openai-mirror/clip-vit-base-patch32

  - teacher: base_model_dir 使用的模型是 Qwen/Qwen3-8B

  - mm_sft: base_model_dir 使用的模型是 Qwen/Qwen3-8B

- src/mpse_mvp/features/video_features.py 内包含的路径有：
  - model_path 使用的路径是 mediapipe/face_landmarker.task

# 如果遇到了读取不出来模块的问题，先下载一下项目:
```bash
python -m pip install -e .
```

# 如何去运行整个流程:
- 原始视频数据要求：在拿到视频文件后，把机器人说的部分给剪去，只留下人说话的部分，把视频文件命名为S*，我们现在已经到了S0019，所以你可以从S0020开始命名，我们现在以S0020为例。把S0020放置在data/raw/中

- 更改config/default.yaml里的配置：session id更改为S0020，pipeline中的前五个开关（run_extract_wav，run_build_turns，run_build_mpse_npz，run_upgrade，run_train_mpse）更改为true，后三个开关（run_build_sft，run_mm_cache，run_train_mm_sft）更改为false。然后运行：
```bash
python scripts/run_full_loop.py --config configs/default.yaml
```
之后我们就会得到outputs文件夹中的部分文件，我们只修改outputs/upgrade中的jsonl文件，我们在该文件内的每一个turn的最后"therapist_reply"部分手动加上理疗师的回复，我们有现成的转录过的pdf文件，直接对照着复制粘贴就可以了。把理疗师的话加上之后，我们关闭前五个开关（run_extract_wav，run_build_turns，run_build_mpse_npz，run_upgrade，run_train_mpse），因为此时原始数据已经处理好了，已经得到了升级后的数据upgrade，接着打开run_build_sft，run_mm_cache，但是run_train_mm_sft要保持关闭因为这个是最终训练开关。然后再次运行：
```bash
python scripts/run_full_loop.py --config configs/default.yaml
```
之后我们就得到了outputs/mm_cache以及sft文件，mm_cache正是我们最终用于训练的数据。我们先把所有的训练数据合并到一起：
```bash
python scripts/merge_mm_index.py
```
得到的是outputs/mm_cache/ALL文件。然后就到了最后训练一步了，把pipeline中除了run_train_mm_sft以外的开关全部关闭，只打开run_train_mm_sft。训练完后我们会得到outputs/mm_sft/ALL，里面包含了lora微调后的权重以及用于三模态对齐的权重mm_prefix.pt

