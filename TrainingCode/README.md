# 高爾夫球 VLM 訓練腳本

本目錄包含從 `FineTune_0513Golf_Data_Gemma3-4b.ipynb` 整理出來的訓練流程，已經模組化成獨立的 Python 腳本。

## 文件結構

```
TrainingCode/
├── train_gemma3_golf.py    # 主要訓練類別
├── config.py               # 配置文件
├── run_training.py         # 執行腳本
└── README.md              # 說明文件
```

## 主要功能

### 1. `train_gemma3_golf.py`
主要的訓練器類別，包含：
- `GolfDatasetTrainer` 類別：處理完整的訓練流程
- 數據載入和預處理
- 模型設置和配置
- 訓練執行和模型保存

### 2. `config.py`
訓練配置管理：
- `TrainingConfig` 類別：統一管理所有訓練參數
- 支持兩種數據類型：`textonly` 和 `hitdata`
- 包含模型、訓練、LoRA 等各種配置

### 3. `run_training.py`
命令行執行腳本，支持參數自定義

## 使用方法

### 方法一：直接使用訓練器

```python
from train_gemma3_golf import GolfDatasetTrainer

# 使用默認配置
trainer = GolfDatasetTrainer()
output_dir = trainer.train()
print(f"模型已保存至: {output_dir}")
```

### 方法二：使用自定義配置

```python
from train_gemma3_golf import GolfDatasetTrainer
from config import TrainingConfig

# 創建配置
config = TrainingConfig()
config.data_type = "hitdata"  # 使用圖文對數據
config.num_train_epochs = 10
config.learning_rate = 2e-5

# 使用配置文件創建訓練器
trainer = GolfDatasetTrainer()
trainer.config = config
output_dir = trainer.train()
```

### 方法三：命令行執行

```bash
# 使用默認配置
python run_training.py

# 自定義參數
python run_training.py \
    --data_type hitdata \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --lora_r 32

# 查看所有可用參數
python run_training.py --help
```

## 配置參數說明

### 數據配置
- `data_type`: 數據類型
  - `textonly`: 純文字問答數據
  - `hitdata`: 圖文對數據
- `file_locate`: 數據文件根目錄路徑

### 訓練配置
- `num_train_epochs`: 訓練週期數 (默認: 15)
- `per_device_train_batch_size`: 批次大小 (默認: 1)
- `learning_rate`: 學習率 (默認: 1e-5)
- `gradient_accumulation_steps`: 梯度累積步數 (默認: 4)

### LoRA 配置
- `lora_r`: LoRA rank (默認: 16)
- `lora_alpha`: LoRA alpha (默認: 16)
- `lora_dropout`: LoRA dropout (默認: 0.05)

### 監控配置
- `wandb_project`: Wandb 專案名稱
- `use_wandb`: 是否使用 wandb (默認: True)

## 數據路徑

腳本會自動根據 `data_type` 載入對應的數據：

- `textonly`: `dataset/0513_SFTDataset/text/qa_pairs_sft.json`
- `hitdata`: `dataset/0513_SFTDataset/hitdata/sft_training_data.json`

## 輸出

訓練完成後，模型會保存在 `model/` 目錄下，文件名格式為：
```
model/{exp_name}{timestamp}/
```

## 注意事項

1. 確保 GPU 支持 bfloat16
2. 確保有足夠的 GPU 記憶體
3. 如果使用 `hitdata` 類型，會自動處理 base64 圖片轉換
4. 訓練過程中會自動清理記憶體
5. 需要先登入 wandb (如果啟用的話)

## 依賴項

```
torch
transformers
trl
peft
wandb
pillow
pandas
```

確保已安裝所有必要的依賴項。 