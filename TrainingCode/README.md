# 高爾夫球 VLM 訓練腳本

本目錄包含從 `FineTune_0513Golf_Data_Gemma3-4b.ipynb` 整理出來的訓練流程，已經模組化成獨立的 Python 腳本。

## 文件結構

```
TrainingCode/
├── train_gemma3_golf.py    # 主要訓練類別
├── config.py               # 配置文件
├── run_training.py         # 執行腳本
├── evaluate.py             # 模型評估腳本
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
- 支持三種數據類型：`textonly`、`hitdata` 和 `mergedata`
- 包含模型、訓練、LoRA 等各種配置

### 3. `evaluate.py`
模型評估器類別，包含：
- `GolfModelEvaluator` 類別：處理完整的評估流程
- 載入已訓練的模型
- 自動執行測試並生成 CSV 結果報告
- 支援純文字和圖文對數據的評估

### 4. `run_training.py`
命令行執行腳本，支持參數自定義並包含自動評估功能

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

**方式A：直接修改 config.py 文件**
```python
# 在 config.py 中修改
class TrainingConfig:
    def __init__(self):
        # ...
        self.data_type = "mergedata"        # 修改數據類型
        self.num_train_epochs = 10          # 修改訓練週期
        self.learning_rate = 2e-5           # 修改學習率
        self.eval_test_count = 10           # 修改評估樣本數量
        # ...
```

**方式B：程式碼中動態修改配置**
```python
from train_gemma3_golf import GolfDatasetTrainer
from config import TrainingConfig

# 創建配置並修改參數
config = TrainingConfig()
config.data_type = "mergedata"
config.num_train_epochs = 10
config.learning_rate = 2e-5
config.eval_test_count = 10

# 使用自定義配置創建訓練器
trainer = GolfDatasetTrainer()
trainer.config = config
output_dir = trainer.train()
```

### 方法三：命令行執行

```bash
# 使用 config.py 中的默認配置（包含自動評估）
python run_training.py

# 跳過自動評估
python run_training.py --skip_evaluation

# 查看可用參數
python run_training.py --help
```

**注意**：所有訓練和評估參數都在 `config.py` 中設定，`run_training.py` 不接受命令行參數覆蓋。如需修改參數，請直接編輯 `config.py` 文件。

### 方法四：獨立執行評估

```bash
# 評估已訓練的模型
python evaluate.py --model_path model/gemma3-4b-sft-textonly-2025_01_15_1230

# 自定義評估參數
python evaluate.py \
    --model_path model/gemma3-4b-sft-textonly-2025_01_15_1230 \
    --test_count 10 \
    --random_select \
    --output_dir my_evaluation_results

# 查看評估腳本的所有參數
python evaluate.py --help
```

### 方法五：程式碼中使用評估器

```python
from evaluate import GolfModelEvaluator
from config import TrainingConfig

# 創建評估器
config = TrainingConfig()
config.data_type = "textonly"  # 根據您的數據類型設定
evaluator = GolfModelEvaluator(config=config)

# 載入模型並評估
evaluator.load_model("model/gemma3-4b-sft-textonly-2025_01_15_1230")
results = evaluator.evaluate_model(
    test_count=5,
    save_csv=True,
    output_dir="experiment_result",
    random_select=False
)

# 清理記憶體
evaluator.cleanup()
```

## 配置參數說明

### 數據配置
- `data_type`: 數據類型
  - `textonly`: 純文字問答數據
  - `hitdata`: 圖文對數據
  - `mergedata`: 合併的圖文對與文字數據
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

### 評估配置
- `eval_test_count`: 評估測試樣本數量 (默認: 5)
- `eval_random_select`: 隨機選擇評估樣本 (默認: False, 即使用前N筆)
- `eval_output_dir`: 評估結果輸出目錄 (默認: "experiment_result")
- `max_new_tokens`: 模型生成最大token數 (默認: 512)
- `top_p`: Top-p 採樣參數 (默認: 0.95)
- `temperature`: 採樣溫度 (默認: 0.7)

**注意**：`skip_evaluation` 是唯一可以通過命令行控制的參數，用於跳過訓練後的自動評估階段。

## 數據路徑

腳本會自動根據 `data_type` 載入對應的數據：

- `textonly`: `dataset/0513_SFTDataset/text/qa_pairs_sft.json`
- `hitdata`: `dataset/0513_SFTDataset/hitdata/sft_training_data.json`
- `mergedata`: `dataset/0513_SFTDataset/mergedata/merged_dataset.json`

## 輸出

### 訓練輸出
訓練完成後，模型會保存在 `model/` 目錄下，文件名格式為：
```
model/{exp_name}{timestamp}/
```

### 評估輸出
評估完成後，結果會保存為 CSV 文件，包含以下欄位：

| 欄位名稱 | 說明 |
|---------|------|
| `sample_index` | 測試樣本在數據集中的索引 |
| `test_order` | 測試順序 |
| `data_type` | 數據類型（純文字/多模態） |
| `question` | 輸入問題 |
| `model_response` | 模型生成的回答 |
| `ground_truth` | 標準答案 |
| `evaluation_status` | 評估狀態（SUCCESS/ERROR） |
| `error_message` | 錯誤訊息（如有） |
| `response_length` | 模型回答長度 |
| `ground_truth_length` | 標準答案長度 |
| `evaluation_time` | 評估時間 |

CSV 文件會保存在指定的輸出目錄中，文件名格式為：
```
model_evaluation_result_{timestamp}.csv
```

## 注意事項

1. 確保 GPU 支持 bfloat16
2. 確保有足夠的 GPU 記憶體
3. 如果使用 `hitdata` 或 `mergedata` 類型，會自動處理 base64 圖片轉換
4. 訓練過程中會自動清理記憶體
5. 需要先登入 wandb (如果啟用的話)
6. 訓練完成後默認會自動執行評估，可以使用 `--skip_evaluation` 跳過
7. 評估階段會重新載入模型，需要確保有足夠記憶體
8. 評估結果會自動保存為 CSV 文件，便於後續分析

## 依賴項

```
torch
transformers
trl
peft
wandb
pillow
pandas
numpy
base64
json
random
datetime
```

確保已安裝所有必要的依賴項。

## 快速開始

1. **安裝依賴項**：
   ```bash
   pip install torch transformers trl peft wandb pillow pandas numpy
   ```

2. **設定參數**：
   編輯 `config.py` 文件來設定您的訓練參數：
   ```python
   # 在 config.py 中修改參數
   self.data_type = "textonly"          # 數據類型
   self.num_train_epochs = 5            # 訓練週期
   self.file_locate = "您的數據路徑/"    # 數據根目錄
   self.eval_test_count = 5             # 評估樣本數量
   ```

3. **執行訓練和評估**：
   ```bash
   python run_training.py
   ```

4. **查看評估結果**：
   訓練完成後會在 `experiment_result/` 目錄下生成 CSV 評估報告 