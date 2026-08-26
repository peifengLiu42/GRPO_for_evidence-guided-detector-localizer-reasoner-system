# 配置目录

该目录用于放置从零 SFT、RFT、GRPO、奖励函数和评测相关配置。

主 SFT 配置为 `qwen3vl4b_reference_sft_from_scratch_distmix_authdown_shortprompt.yaml`。
该配置从 Qwen3-VL-4B 开始训练，训练 5 个 epoch 的新 LoRA adapter，并打开最终
RFT + ms-swift GRPO 流程所需的语言模型、视觉编码器和多模态投影层路径。
