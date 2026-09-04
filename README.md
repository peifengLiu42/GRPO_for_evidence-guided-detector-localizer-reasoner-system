# 面向证据引导伪造检测定位推理系统的 GRPO 训练代码

本仓库整理了一个面向文档伪造检测与定位的证据引导式视觉语言推理系统，包含从零 SFT、RFT 和基于 ms-swift 的 GRPO 训练与评测代码。核心思路是把小模型检测器和定位器的预测结果作为可能有误的参考证据，再训练视觉语言推理模型去验证、修正、拒绝或补充这些证据。

## 方法概览

训练流程：

```text
检测器 + 定位器预测
        |
        v
构造参考证据 prompt
        |
        v
基于 GT 取证报告从零 SFT
        |
        v
基于规则过滤高质量回答进行 RFT
        |
        v
使用定位对齐奖励进行 GRPO
```

检测器和定位器预测只放入 prompt。真实标签、真实框、真实 mask 和真实报告只用于训练目标、采样元信息或奖励计算，不会泄露到用户输入中。

## 主要结果

在 RealTextV2 域内验证集上，改进后的 GRPO 奖励在保持较高图像级准确性的同时，提升了像素级定位质量：

| model | image Bal-Acc | image Wtd-F1 | pixel P | pixel R | pixel F1 | pixel IoU |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 99.38 | 99.41 | 45.30 | 73.27 | 49.44 | 36.83 |
| RFT | 99.40 | 99.41 | 52.57 | 71.90 | 54.84 | 42.13 |
| GRPO ckpt-500 | 99.57 | 99.56 | 53.84 | 70.82 | 55.54 | 42.75 |
| GRPO ckpt-1000 | 99.40 | 99.41 | 58.38 | 70.85 | 58.60 | 45.82 |
| GRPO ckpt-1500 | 99.40 | 99.41 | 57.23 | 72.23 | 58.15 | 45.37 |

其中 SFT 为同一评测脚本复算口径；原始记录中的 SFT official 结果为 image Bal-Acc 99.38、image Wtd-F1 99.41、pixel P 45.77、pixel R 73.95、pixel F1 49.92、pixel IoU 37.41。

## 奖励函数设计

GRPO 奖励函数实现在 `src/realtext_grpo/msswift_reward_plugin.py` 和 `src/realtext_grpo/rewards.py`。论文公式化说明见 `docs/reward_design.md`。

奖励函数主要包含：

- 输出格式奖励；
- 图像级分类正确性；
- 基于 IoU 阈值 0.3 的一对一匹配 Box F1；
- 集合级 Set IoU；
- 从 IoU 0.5 开始计算的高 IoU bonus；
- 像素级 precision、recall 和 IoU；
- 伪造样本预测区域过大的过框惩罚；
- 真实样本上输出伪造框的误检惩罚。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── qwen3vl4b_reference_sft_from_scratch_distmix_authdown_shortprompt.yaml
│   └── qwen3vl4b_reference_rft_from_sft5_distmix_authdown_shortprompt.yaml
├── data/
│   ├── README.md
│   ├── dataset_info.json
│   └── *.stats.json
├── src/realtext_grpo/
│   ├── prompts.py
│   ├── dataset.py
│   ├── rewards.py
│   └── msswift_reward_plugin.py
├── scripts/
│   ├── data/
│   ├── train/
│   ├── eval/
│   └── monitor/
├── dataset/
│   └── generate_realtext_grpo_pred_evidence.py
└── docs/
    ├── data_configuration.md
    ├── reward_design.md
    └── realtext_from_scratch_sft_grpo_predicted_evidence_plan.md
```

仓库中不包含大文件产物：原始图像、mask、生成后的训练 JSON/JSONL、模型 checkpoint、合并模型、vLLM 预测结果和训练曲线。

## 环境配置

推荐环境：

```bash
conda create -n realtext_msswift python=3.10 -y
conda activate realtext_msswift
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

SFT/RFT 使用 LLaMA-Factory，可按需单独准备环境。GRPO 使用 ModelScope SWIFT，需要安装 Qwen3-VL 和 RLHF 相关依赖。

## 数据输入

需要提前准备以下输入：

```text
RealTextV2 训练图像
RealTextV2 GT mask
RealTextV2 验证/测试划分列表
带有 P(forged) 的检测器预测结果
DTD/定位器概率图或二值 mask
包含 GT 取证报告的原始 SFT 文件
Qwen3-VL 基座模型
```

常用环境变量：

```bash
export QWEN3_VL_MODEL=/path/to/Qwen3-VL-4B-Instruct
export REALTEXT_SOURCE_SFT=/path/to/realtext_train_sft.json
export REALTEXT_IMAGE_ROOT=/path/to/RealTextV2/train/image
export REALTEXT_GT_MASK_ROOT=/path/to/RealTextV2/train/regen_mask
export REALTEXT_EXCLUDE_LIST=/path/to/RealTextV2/train/test.txt
export REALTEXT_DTD_MASK_DIR=/path/to/dtd_masks
export REALTEXT_DETECTOR_JSON=/path/to/detector_predictions.jsonl
```

大多数脚本也提供等价的命令行参数。

SFT、RFT 和 GRPO 的详细数据配置见 `docs/data_configuration.md`。

## 构造参考证据数据

生成完整候选池：

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

生成从零 SFT 子集：

```bash
python scripts/data/generate_from_scratch_sft_data.py \
  --input data/realtext_grpo_reference_evidence_train_shortprompt.json \
  --num_samples 7851 \
  --forged_fraction 0.586931600 \
  --output data/realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.json \
  --stats_output data/realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.stats.json \
  --overwrite
```

使用 ultra-hard 采样将 GRPO 数据转换为 ms-swift JSONL：

```bash
python scripts/data/convert_to_msswift.py \
  --input data/realtext_grpo_reference_evidence_train_shortprompt.json \
  --output data/realtext_grpo_reference_evidence_train_shortprompt_msswift_ultrahardmix3k_seed42.jsonl \
  --sampling_mode ultra_hardmix \
  --num_records 3000 \
  --seed 42
```

## SFT

修改以下配置中的路径：

```text
configs/qwen3vl4b_reference_sft_from_scratch_distmix_authdown_shortprompt.yaml
```

启动训练：

```bash
CONDA_ENV=llama-factory \
GPUS=0,1,2,3 \
CONFIG=configs/qwen3vl4b_reference_sft_from_scratch_distmix_authdown_shortprompt.yaml \
bash scripts/train/run_sft.sh
```

主 SFT 配置训练 rank-32 LoRA adapter，语言模型通过 LoRA 参与训练，同时打开视觉编码器/投影层路径。

## RFT

使用 vLLM 生成模型回答：

```bash
python scripts/eval/rollout_rft_vllm.py \
  --model_name_or_path /path/to/sft_merged_model \
  --source_json data/realtext_grpo_reference_evidence_train_shortprompt.json \
  --output_jsonl outputs/rft_rollouts.jsonl \
  --overwrite
```

通过规则过滤构造 RFT 数据：

```bash
python scripts/data/build_rft_data_from_rollouts.py \
  outputs/rft_rollouts.jsonl \
  --output_json data/realtext_reference_rft_from_sft5_distmix_authdown_shortprompt.json \
  --stats_output data/realtext_reference_rft_from_sft5_distmix_authdown_shortprompt.stats.json
```

随后使用提供的 RFT YAML 配置继续训练。

## 基于 ms-swift 的 GRPO

GRPO 主入口：

```text
scripts/train/run_msswift_grpo_5epoch_eval500.sh
```

启动示例：

```bash
tmux new-session -d -s realtext_grpo \
  "cd $PWD && \
   SWIFT_SINGLE_DEVICE_MODE=1 \
   TRAIN_GPUS=0,1,2,3 \
   EVAL_GPUS=0,1,2,3 \
   TOTAL_EPOCHS=5 \
   EVAL_INTERVAL=500 \
   NUM_RECORDS=3000 \
   NUM_GENERATIONS=8 \
   PER_DEVICE_TRAIN_BATCH_SIZE=4 \
   LEARNING_RATE=2e-6 \
   BETA=0.05 \
   TEMPERATURE=0.9 \
   TOP_P=0.95 \
   TOP_K=50 \
   bash scripts/train/run_msswift_grpo_5epoch_eval500.sh"
```

常用监控命令：

```bash
tmux attach -t realtext_grpo
tail -f outputs/logs/msswift_grpo_run2_5epoch_eval500_gen8_train.log
nvidia-smi --query-compute-apps=pid,process_name,gpu_bus_id,used_memory --format=csv,noheader,nounits | sort -n
```

`SWIFT_SINGLE_DEVICE_MODE=1` 会让每个分布式 rank 只看到一张物理 GPU，从而得到更清晰的一卡一进程视图。

## 评测

运行 vLLM 推理：

```bash
python scripts/eval/infer_vllm.py \
  --evidence_jsonl data/realtext_indomain_reference_evidence.jsonl \
  --model_name_or_path /path/to/merged_model \
  --merged_model \
  --output_jsonl outputs/realtext_indomain_vllm.jsonl \
  --overwrite
```

计算图像级和像素级指标：

```bash
python scripts/eval/evaluate_image_pixel_metrics.py \
  outputs/realtext_indomain_vllm.jsonl \
  --gt_mask_root /path/to/RealTextV2/train/regen_mask \
  --mask_dir outputs/masks \
  --output_json outputs/image_pixel_metrics.json
```

## 注意事项

- 不要把 GT 标签或 GT 框放入用户 prompt。
- SFT、RFT 和 GRPO 训练数据都需要排除验证/测试集样本。
- GRPO 优先使用 hard 或 ultra-hard 采样，让 rollout group 具有足够的奖励方差。
- 评测时同时关注图像级指标和像素级定位指标，不能只看 reward 是否上升。
