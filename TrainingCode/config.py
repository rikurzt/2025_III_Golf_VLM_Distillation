#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高爾夫球 VLM 訓練配置文件
"""

import os
from typing import Dict, Any

class TrainingConfig:
    """訓練配置類別"""
    
    def __init__(self):
        # 實驗配置
        self.exp_name = "gemma3-4b-sft-hitdata-"
        self.file_locate = "/tmp/pycharm_project_979/"  # 遠端環境，根據部署地做更改
        
        # 數據配置
        # textonly: 對應 dataset/0513_SFTDataset/text/qa_pairs_sft.json，只包含文字問題
        # hitdata: 對應 dataset/0513_SFTDataset/hitdata/sft_training_data.json，為圖文對資料
        self.data_type = "hitdata"  # textonly 或 hitdata
        
        # 模型配置
        self.model_id = "google/gemma-3-27b-pt"
        self.processor_id = "google/gemma-3-27b-it"
        
        # 訓練配置
        self.num_train_epochs = 5                      # 訓練週期數
        self.per_device_train_batch_size = 1            # 每個設備的批次大小
        self.gradient_accumulation_steps = 4            # 梯度累積步數
        self.learning_rate = 2e-4                       # 學習率
        self.max_grad_norm = 0.3                        # 最大梯度範數
        self.warmup_ratio = 0.03                        # 暖身比例
        self.lr_scheduler_type = "constant"             # 學習率調度器類型
        
        # LoRA配置
        self.lora_alpha = 16                            # LoRA alpha 參數
        self.lora_dropout = 0.05                        # LoRA dropout 比例
        self.lora_r = 16                                # LoRA rank
        
        # 日誌和監控配置
        self.wandb_project = "III-2025-golf"            # Wandb 專案名稱
        self.use_wandb = True                           # 是否使用 wandb
        self.logging_steps = 1                          # 日誌記錄步數
        
        # 模型技術配置
        self.torch_dtype = "bfloat16"                   # Torch 數據類型
        self.attn_implementation = "eager"              # 注意力實現方式
        self.device_map = "auto"                        # 設備映射
        
        # 圖片處理配置
        self.image_scale_percent = 20                   # 圖片縮放比例（百分比）
        self.image_quality = 50                         # 圖片壓縮品質
        
        # 生成配置（用於評估）
        self.max_new_tokens = 512                       # 最大生成token數
        self.top_p = 0.95                               # top-p 採樣
        self.temperature = 0.7                          # 採樣溫度
        
    def update_config(self, **kwargs):
        """更新配置參數"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"更新配置: {key} = {value}")
            else:
                print(f"警告: 未知的配置參數 {key}")
    
    def get_data_path(self):
        """根據數據類型獲取數據路徑"""
        if self.data_type == "textonly":
            return f"{self.file_locate}dataset/0513_SFTDataset/text/qa_pairs_sft.json"
        elif self.data_type == "hitdata":
            return f"{self.file_locate}dataset/0513_SFTDataset/hitdata/sft_training_data.json"
        else:
            raise ValueError(f"未知的數據類型: {self.data_type}")

    def print_config(self):
        """打印當前配置"""
        print("=" * 50)
        print("訓練配置:")
        print(f"  實驗名稱: {self.exp_name}")
        print(f"  數據類型: {self.data_type}")
        print(f"  模型ID: {self.model_id}")
        print(f"  訓練週期: {self.num_train_epochs}")
        print(f"  批次大小: {self.per_device_train_batch_size}")
        print(f"  學習率: {self.learning_rate}")
        print(f"  LoRA rank: {self.lora_r}")
        print(f"  注意力實現: {self.attn_implementation}")
        print(f"  Torch 數據類型: {self.torch_dtype}")
        print(f"  Wandb 專案: {self.wandb_project}")
        print("=" * 50)


# 預設配置實例
default_config = TrainingConfig() 