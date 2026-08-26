# SFT/RFT/GRPO 数据配置

本文档说明最终代码仓库中三阶段数据的来源、采样方式和配置比例。完整流程为：

```text
完整参考证据候选池
        |
        +--> 从零 SFT 数据
        |
        +--> SFT 模型 rollout
                 |
                 +--> 规则过滤 RFT 数据
        |
        +--> ultra-hard GRPO 数据
```

所有阶段都使用同一套参考证据 prompt：用户输入中只包含原图、检测器预测和定位器预测框。真实标签、真实框、真实 mask 和真实报告不进入用户 prompt，只用于 SFT/RFT 的训练目标、数据采样统计或 GRPO 奖励计算。

## 1. 完整参考证据候选池

候选池由 `scripts/data/generate_reference_evidence_data.py` 生成：

```bash
python scripts/data/generate_reference_evidence_data.py \
  --source_sft /path/to/realtext_train_sft.json \
  --image_root /path/to/RealTextV2/train/image \
  --gt_mask_root /path/to/RealTextV2/train/regen_mask \
  --exclude_list /path/to/RealTextV2/train/test.txt \
  --dtd_mask_dir /path/to/dtd_masks \
  --detector_json /path/to/detector_predictions.jsonl \
  --output data/realtext_grpo_reference_evidence_train_shortprompt.json
```

候选池规模为 `12148` 条，其中：

| 类别 | 数量 |
|---|---:|
| FORGED | 6748 |
| AUTHENTIC | 5400 |

生成时会排除验证/测试列表中的样本，最终 `validation_overlap=0`。每条样本保留三部分：

- `messages`：ShareGPT 格式的 system/user/assistant；assistant 是 GT 取证报告；
- `images`：图像路径；
- `grpo_metadata`：只用于采样和奖励计算的元信息，不作为用户输入。

候选池中根据定位证据质量划分 difficulty bucket：

| bucket | 含义 | 源候选池数量 |
|---|---|---:|
| `forged_iou_lt_0.3` | forged 样本，DTD 框与 GT IoU < 0.3 | 636 |
| `forged_iou_0.3_0.7` | forged 样本，0.3 <= DTD 框与 GT IoU < 0.7 | 1984 |
| `forged_iou_ge_0.7` | forged 样本，DTD 框与 GT IoU >= 0.7 | 4113 |
| `forged_dtd_empty` | forged 样本，但定位器没有给出框 | 15 |
| `authentic_dtd_fp` | authentic 样本，但定位器给出了可疑框 | 243 |
| `authentic_clean` | authentic 样本，定位器没有给出框 | 5157 |

同时给每条样本设置推荐采样权重：

| 权重 | 含义 | 源候选池数量 |
|---:|---|---:|
| 1 | 普通样本 | 9251 |
| 2 | 中等难度定位样本 | 1969 |
| 4 | 定位困难、真实误检、检测器错误或检测器/定位器冲突样本 | 928 |

权重采用 max 规则，而不是多个因素相乘，避免极少数样本被过度重复。

## 2. 从零 SFT 数据配置

SFT 数据由 `scripts/data/generate_from_scratch_sft_data.py` 从完整候选池中抽取：

```bash
python scripts/data/generate_from_scratch_sft_data.py \
  --input data/realtext_grpo_reference_evidence_train_shortprompt.json \
  --num_samples 7851 \
  --forged_fraction 0.586931600 \
  --seed 42 \
  --output data/realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.json \
  --stats_output data/realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.stats.json \
  --overwrite
```

SFT 目标不是只训练“证据完全正确”的简单样本，而是让模型从一开始就知道参考证据可能有误。因此 SFT 保留源数据中的证据质量分布，同时降低 authentic clean 的占比，强制包含检测器错误或检测器/定位器冲突样本。

SFT 总量为 `7851` 条：

| 类别 | 数量 | 占比 |
|---|---:|---:|
| FORGED | 4608 | 58.69% |
| AUTHENTIC | 3243 | 41.31% |

SFT difficulty bucket 配比：

| bucket | 数量 |
|---|---:|
| `forged_iou_lt_0.3` | 433 |
| `forged_iou_0.3_0.7` | 1353 |
| `forged_iou_ge_0.7` | 2807 |
| `forged_dtd_empty` | 15 |
| `authentic_dtd_fp` | 243 |
| `authentic_clean` | 3000 |

forged 样本的 IoU 分布如下：

| DTD-GT IoU 区间 | 数量 |
|---|---:|
| `[0.0,0.1)` | 95 |
| `[0.1,0.2)` | 149 |
| `[0.2,0.3)` | 204 |
| `[0.3,0.4)` | 246 |
| `[0.4,0.5)` | 266 |
| `[0.5,0.6)` | 375 |
| `[0.6,0.7)` | 466 |
| `[0.7,0.8)` | 611 |
| `[0.8,0.9)` | 986 |
| `[0.9,1.0)` | 1210 |

SFT 的训练目标是原始 GT 取证报告，所以它提供的是“标准答案模仿能力”：模型学习输出格式、图像级判断、风险分数、摘要和 grounding 框。由于 prompt 中证据可能错误，模型也会学习在证据不可靠时修正或忽略参考框。

## 3. RFT 数据采集与过滤

RFT 的目标是用 SFT 模型自己生成的高质量回答继续监督训练，使训练目标更贴近模型自己的输出分布。流程分两步：

```text
SFT merged model
        |
        v
vLLM 多回答 rollout
        |
        v
规则打分与过滤
        |
        v
RFT ShareGPT 数据
```

### 3.1 rollout 采集

使用 `scripts/eval/rollout_rft_vllm.py` 从 SFT5 merged 模型采样回答：

```bash
python scripts/eval/rollout_rft_vllm.py \
  --model_name_or_path outputs/qwen3vl4b_reference_sft5_final_visionlora_projector_merged_bf16 \
  --source_json data/realtext_grpo_reference_evidence_train_shortprompt.json \
  --output_jsonl outputs/rft_sft5_rollouts/rollouts.part0.jsonl \
  --max_prompts 4096 \
  --forged_fraction 0.6 \
  --n 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_new_tokens 2048 \
  --resize 1280 \
  --max_pixels 1048576 \
  --overwrite
```

采集配置：

| 参数 | 值 | 说明 |
|---|---:|---|
| `max_prompts` | 4096 | 用于 RFT rollout 的 prompt 数 |
| `forged_fraction` | 0.6 | forged prompt 约占 60% |
| `n` | 4 | 每个 prompt 生成 4 个候选回答 |
| `temperature` | 0.7 | 保持一定多样性 |
| `top_p` | 0.9 | nucleus sampling |
| `max_new_tokens` | 2048 | 保证完整报告和 grounding 输出 |

prompt 选择时按类别定额，并使用 `recommended_sampling_weight` 加权无放回抽样，因此困难样本更容易被采到。

### 3.2 RFT 规则过滤

使用 `scripts/data/build_rft_data_from_rollouts.py` 对 rollout 结果打分和过滤：

```bash
python scripts/data/build_rft_data_from_rollouts.py \
  outputs/rft_sft5_rollouts/rollouts.part*.jsonl \
  outputs/rft_sft5_rollouts/rollouts.resume8.part*.jsonl \
  --output_json data/realtext_reference_rft_from_sft5_distmix_authdown_shortprompt.json \
  --stats_output data/realtext_reference_rft_from_sft5_distmix_authdown_shortprompt.stats.json \
  --details_jsonl outputs/rft_sft5_rollouts/filtered_details.jsonl \
  --overwrite
```

过滤逻辑是：对每个 prompt 的多个候选回答分别解析类别、格式和 grounding 框，再计算像素级质量；选择综合分最高的候选。如果该候选通过 hard-pass 条件，就进入 RFT 数据。

默认过滤阈值：

| 参数 | 值 | 作用 |
|---|---:|---|
| `min_score` | 0.40 | 最低综合分 |
| `min_forged_iou` | 0.02 | forged 样本至少要和 GT 有非零区域重叠 |
| `min_forged_precision` | 0.05 | forged 样本预测框不能几乎全是误检 |
| `max_forged_area_ratio` | 0.35 | forged 样本预测区域不能覆盖过大 |
| `max_authentic_boxes` | 0 | authentic 样本不允许输出 grounding 框 |

hard-pass 条件：

- 格式奖励 `format >= 0.75`；
- 图像级类别必须正确；
- 不允许非法框；
- authentic 样本必须预测为 `AUTHENTIC` 且无框；
- forged 样本必须预测为 `FORGED`，至少有一个有效框，并满足 IoU、precision 和面积比例阈值。

forged 回答的综合分为：

\[
S =
0.20R_{\mathrm{fmt}}
+0.20C
+0.25F1_{\mathrm{pix}}
+0.25IoU_{\mathrm{pix}}
+0.10P_{\mathrm{pix}}
-2\max(0,A_{\mathrm{pred}}-0.10),
\]

其中 \(C\) 表示分类是否正确，\(A_{\mathrm{pred}}\) 是预测区域占整图面积比例。

authentic 回答的综合分为：

\[
S =
0.20R_{\mathrm{fmt}}
+0.30C
+0.50Q_{\mathrm{clean}},
\]

其中 \(Q_{\mathrm{clean}}=1\) 表示模型预测为 authentic 且没有输出框，否则为 0。

最终 RFT 数据统计：

| 项目 | 数值 |
|---|---:|
| rollout prompt 数 | 4096 |
| RFT 接收样本数 | 4050 |
| 接收率 | 98.88% |
| accepted forged | 2413 |
| accepted authentic | 1637 |
| rejected forged | 45 |
| rejected authentic | 1 |

进入 RFT 的回答平均像素质量：

| 指标 | 数值 |
|---|---:|
| precision | 71.46% |
| recall | 80.46% |
| F1 | 71.87% |
| IoU | 63.70% |

RFT 训练时使用过滤后的回答作为 assistant 监督信号，不把 `grpo_metadata` 写入训练 JSON。

## 4. GRPO 数据配置

GRPO 不使用 assistant 标准答案做监督，而是把 prompt-only 数据交给 ms-swift。模型在线生成 \(G\) 个回答，由奖励函数根据 GT 标签/框/mask 计算奖励。

转换脚本为 `scripts/data/convert_to_msswift.py`：

```bash
python scripts/data/convert_to_msswift.py \
  --input data/realtext_grpo_reference_evidence_train_shortprompt.json \
  --output data/realtext_grpo_reference_evidence_train_shortprompt_msswift_ultrahardmix3k_seed42.jsonl \
  --sampling_mode ultra_hardmix \
  --num_records 3000 \
  --seed 42
```

ms-swift JSONL 中保留：

- `messages`：只有 system/user，没有 assistant；
- `images`：图像路径；
- `gt_label`、`gt_boxes`、`gt_risk`：奖励函数使用；
- `reference_answer`：仅用于调试或奖励侧参考，不作为监督答案；
- `difficulty_bucket`、`evidence_case`、`sampling_weight`：采样和统计使用。

最终 GRPO 使用 `ultra_hardmix`，每阶段 3000 条，配比如下：

| bucket | fraction | count / 3000 |
|---|---:|---:|
| `forged_iou_lt_0.3` | 0.33 | 990 |
| `forged_iou_0.3_0.7` | 0.28 | 840 |
| `forged_iou_ge_0.7` | 0.05 | 150 |
| `forged_dtd_empty` | 0.02 | 60 |
| `authentic_dtd_fp` | 0.25 | 750 |
| `authentic_clean` | 0.07 | 210 |

类别比例：

| 类别 | 数量 | 占比 |
|---|---:|---:|
| FORGED | 2040 | 68% |
| AUTHENTIC | 960 | 32% |

这个配比刻意减少简单样本：

- `authentic_clean` 很容易拿满 reward，组内优势方差小，所以只保留 7%；
- `forged_iou_ge_0.7` 证据已经较准，模型主要需要复用，训练价值较低，所以降到 5%；
- `forged_iou_lt_0.3`、`forged_iou_0.3_0.7` 和 `authentic_dtd_fp` 是主要困难来源，能提供更强的奖励差异。

## 5. GRPO 动态数据机制

正式训练入口 `scripts/train/run_msswift_grpo_5epoch_eval500.sh` 默认开启动态数据：

```text
DYNAMIC_DATA=1
DYNAMIC_SAMPLING_MODE=ultra_hardmix
NUM_RECORDS=3000
DATA_SEED_BASE=42
EVAL_INTERVAL=500
```

它的含义是：每训练到一个新的 500-step stage，就用相同的 `ultra_hardmix` 比例重新抽一份 3000 条数据，但 seed 会变化：

\[
\mathrm{seed}_{stage} = 42 + stage\_index.
\]

例如：

| 训练目标 step | stage index | seed | 数据 |
|---:|---:|---:|---|
| 500 | 0 | 42 | stage0000 |
| 1000 | 1 | 43 | stage0001 |
| 1500 | 2 | 44 | stage0002 |

这样做的目的不是改变数据分布，而是在保持 hard 配比稳定的同时，让模型每个阶段看到不同的困难样本组合，减少固定 3000 条数据被反复过拟合。

## 6. 三阶段数据对比

| 阶段 | 数据来源 | 是否有 assistant 监督 | 样本数 | 主要目标 |
|---|---|---|---:|---|
| SFT | 完整候选池抽样 + GT 报告 | 有，GT 报告 | 7851 | 学会格式、基础判断和 grounding 表达 |
| RFT | SFT 模型 rollout 后规则过滤 | 有，模型高质量回答 | 4050 | 让监督目标贴近模型输出分布，提高像素定位质量 |
| GRPO | 完整候选池 ultra-hard prompt-only 抽样 | 无，在线奖励 | 3000/阶段 | 用奖励函数直接优化分类、框匹配和像素级定位 |

简而言之，SFT 负责打底，RFT 用规则筛出的好回答修正输出分布，GRPO 则把训练重心集中到困难证据样本上，用奖励函数继续拉开好框和坏框的差距。
