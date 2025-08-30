# 高爾夫球 VLM 訓練/蒸餾腳本

本目錄提供高爾夫球影像-文字模型的微調、LoRA 合併、教師訊號生成與蒸餾訓練，以及評估的完整工具鏈。

## 目錄結構

```
TrainingCode/
├── finetune_gemma3_golf.py      # 微調訓練器 (SFT)
├── distill_gemma3_golf.py       # 蒸餾訓練器 (SFT + KD)
├── distillation.py              # 蒸餾資料集/損失/Collator 工具
├── generate_distill_data.py     # 用教師模型產生 teacher signals (CSV)
├── merge_lora.py                # 合併 LoRA 適配器到基礎模型
├── evaluate.py                  # 模型評估
├── run_training.py              # 一鍵指令入口/子指令
├── config.py                    # 集中設定 (分組 ft_* / ds_*)
└── README.md                    # 本說明文件
```

## 功能概覽
- 微調 (SFT)：在 `finetune_gemma3_golf.py` 進行，參數使用 `ft_*` 分組。
- LoRA 合併：`merge_lora.py` 將適配器權重合併為完整模型。
- 教師訊號生成：`generate_distill_data.py` 從合併後教師模型導出 hidden/attn/image_hidden 到 CSV。
- 蒸餾 (SFT+KD)：在 `distill_gemma3_golf.py` 進行，參數使用 `ds_*` 分組。
- 評估：`evaluate.py` 對任一模型做推理並輸出 CSV 報告。
- 一鍵流程：`run_training.py` 可依子指令分步執行，或不帶子指令直接跑全部階段。

## 快速開始

1) 安裝依賴
```bash
pip install -r ../requirements.txt
```

2) 設定資料路徑 (建議 Windows 直接以斜線寫法)
```bash
# 範例：在 CLI 覆寫到專案根目錄
python3 run_training.py --file_locate "D:/WorkSpace/2025_III_Golf_VLM_Distillation/" --data_type mergedata
```

3) 一鍵跑完整流程 (微調 → 合併LoRA → 生成教師訊號 → 蒸餾 → 評估)
```bash
python3 run_training.py --file_locate "D:/WorkSpace/2025_III_Golf_VLM_Distillation/" --data_type mergedata
```

## 子指令與用法

`run_training.py` 支援以下子指令；所有 `config.py` 內欄位都可用 CLI 覆寫（包含布林 `--flag/--no-flag`）。

- 微調
```bash
python3 run_training.py finetune \
  --file_locate "D:/WorkSpace/2025_III_Golf_VLM_Distillation/" \
  --data_type mergedata \
  --ft_model_id google/gemma-3-4b-pt --ft_processor_id google/gemma-3-4b-it \
  --ft_torch_dtype bfloat16 --ft_attn_implementation eager --ft_device_map auto \
  --ft_use_4bit --ft_learning_rate 2e-4 --ft_num_train_epochs 5 \
  --ft_per_device_train_batch_size 1 --ft_gradient_accumulation_steps 4
# 訓練完成後自動評估，如需跳過：
python3 run_training.py finetune --skip_evaluation
```

- 合併 LoRA
```bash
python3 run_training.py merge-lora \
  --lora_adapter_path path/to/lora/checkpoint \
  --base_model_id google/gemma-3-4b-pt \
  --lora_merge_max_shard_size 5GB
```

- 生成教師訊號 (teacher signals)
```bash
python3 run_training.py generate-distill-data \
  --teacher_model_path model/merged_teacher_model \
  --processor_id google/gemma-3-4b-it \
  --file_locate "D:/WorkSpace/2025_III_Golf_VLM_Distillation/" \
  --data_type mergedata
```

- 蒸餾
```bash
python3 run_training.py distill \
  --file_locate "D:/WorkSpace/2025_III_Golf_VLM_Distillation/" \
  --data_type mergedata \
  --teacher_model_path model/merged_teacher_model \
  --ds_student_model_id google/gemma-3-4b-pt --ds_processor_id google/gemma-3-4b-it \
  --ds_torch_dtype bfloat16 --ds_attn_implementation eager --ds_device_map auto \
  --no-ds_use_4bit --ds_learning_rate 1e-4 --ds_num_train_epochs 3 \
  --distill_loss_weight 0.6
# 訓練完成後自動評估，如需跳過：
python3 run_training.py distill --skip_evaluation
```

- 評估
```bash
python3 run_training.py evaluate --model_path path/to/model_dir \
  --file_locate "D:/WorkSpace/2025_III_Golf_VLM_Distillation/" \
  --data_type mergedata \
  --eval_test_count 10 --eval_output_dir experiment_result --eval_random_select
```

- 查看所有可覆寫欄位
```bash
python3 run_training.py --help | cat
```

## 設定說明（關鍵）

- 分組：預設即分開，微調讀 `ft_*`，蒸餾讀 `ds_*`，互不影響。
- 常用共通欄位：
  - `file_locate`：專案/資料根目錄（例如 `D:/WorkSpace/2025_III_Golf_VLM_Distillation/`）
  - `data_type`：`textonly` | `hitdata` | `mergedata`
  - `lora_alpha`, `lora_dropout`, `lora_r`, `lora_merge_max_shard_size`
  - `teacher_model_path`, `distill_loss_weight`, `distill_dataset_locate`, `merged_model_path`
- 微調 (ft_*，預設值見 `config.py`)
  - 模型載入：`ft_model_id`, `ft_processor_id`, `ft_torch_dtype`, `ft_attn_implementation`, `ft_device_map`, `ft_use_4bit`, `ft_bnb_4bit_*`
  - SFT：`ft_num_train_epochs`, `ft_per_device_train_batch_size`, `ft_gradient_accumulation_steps`, `ft_learning_rate`, `ft_max_grad_norm`, `ft_warmup_ratio`, `ft_lr_scheduler_type`, `ft_gradient_checkpointing`, `ft_gradient_checkpointing_use_reentrant`, `ft_optim`, `ft_save_strategy`, `ft_logging_steps`
- 蒸餾 (ds_*，預設值見 `config.py`)
  - 學生模型載入：`ds_student_model_id`, `ds_processor_id`, `ds_torch_dtype`, `ds_attn_implementation`, `ds_device_map`, `ds_use_4bit`, `ds_bnb_4bit_*`
  - SFT (蒸餾訓練)：`ds_num_train_epochs`, `ds_per_device_train_batch_size`, `ds_gradient_accumulation_steps`, `ds_learning_rate`, `ds_max_grad_norm`, `ds_warmup_ratio`, `ds_lr_scheduler_type`, `ds_gradient_checkpointing`, `ds_gradient_checkpointing_use_reentrant`, `ds_optim`, `ds_save_strategy`, `ds_logging_steps`

所有欄位皆可於 `config.py` 直接修改，或在命令列以 `--欄位名 值` 覆寫。

## 數據路徑
依 `data_type` 自動載入：
- `textonly`: `dataset/0513_SFTDataset/text/qa_pairs_sft.json`
- `hitdata`: `dataset/0513_SFTDataset/hitdata/sft_training_data.json`
- `mergedata`: `dataset/0513_SFTDataset/mergedata/merged_dataset.json`

## 輸出
- 訓練輸出：`model/{exp_name}{timestamp}/`
- 評估輸出：`experiment_result/model_evaluation_result_{timestamp}.csv`

## 注意事項
- 建議使用支援 bfloat16 的 GPU；如需 4-bit，請保持 `*_use_4bit` 為開啟並確認環境支援 bitsandbytes。
- `generate-distill-data` 需先有合併後的教師模型 (`merge-lora`)；產生的 CSV 預設放於 `dataset/distill_teacher_signals.csv`。
- Windows 使用者建議 `file_locate` 採用斜線路徑，例如：`D:/WorkSpace/2025_III_Golf_VLM_Distillation/`。
- `run_training.py` 不帶子指令時，會依序執行：微調 → LoRA 合併 → 更新教師模型路徑 → 生成蒸餾資料 → 蒸餾 → 評估。 