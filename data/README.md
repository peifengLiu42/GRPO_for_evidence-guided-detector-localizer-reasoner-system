# 数据目录

该目录用于放置生成后的 manifest 和训练 JSON/JSONL 文件。不要把原始 RealTextV2 图像、mask、checkpoint 或预测 dump 放入本仓库。

GitHub 版本只保留三阶段训练流程对应的轻量元信息：

- `dataset_info.json`：LLaMA-Factory 数据集注册模板；
- `realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.stats.json`：SFT 子集统计；
- `realtext_reference_rft_from_sft5_distmix_authdown_shortprompt.stats.json`：RFT 子集统计；
- `realtext_grpo_reference_evidence_train_shortprompt_msswift_ultrahardmix3k_seed42.jsonl.stats.json`：GRPO ms-swift 采样统计。

主要生成文件已通过 `.gitignore` 忽略：

- `realtext_grpo_reference_evidence_train_shortprompt.json`：用于派生各阶段数据的完整预测证据候选池；
- `realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.json`：SFT 子集；
- `realtext_reference_rft_from_sft5_distmix_authdown_shortprompt.json`：RFT 子集；
- `realtext_grpo_reference_evidence_train_shortprompt_msswift_ultrahardmix3k_seed42.jsonl`：ms-swift GRPO 训练 JSONL。

可根据顶层 `README.md` 中的命令，使用本地路径重新生成数据。
