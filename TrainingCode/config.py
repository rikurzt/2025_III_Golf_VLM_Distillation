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

        self.file_locate = "/tmp/pycharm_project_640/"  # 遠端環境，根據部署地做更改
        
        # 數據配置
        self.data_type = "mergedata"

        # 共用基礎（資料與命名）
        self.model_id = "google/gemma-3-27b-pt"
        self.processor_id = "google/gemma-3-27b-it"
        self.exp_name = self.model_id + self.data_type + "-"

        # —— 微調分組 (ft_*) 預設 ——
        # 模型載入
        self.ft_model_id = self.model_id
        self.ft_processor_id = self.processor_id
        self.ft_torch_dtype = "bfloat16"
        self.ft_attn_implementation = "eager"
        self.ft_device_map = "auto"
        self.ft_use_4bit = True
        self.ft_bnb_4bit_use_double_quant = True
        self.ft_bnb_4bit_quant_type = "nf4"
        # SFT
        self.ft_num_train_epochs = 5
        self.ft_per_device_train_batch_size = 1
        self.ft_gradient_accumulation_steps = 4
        self.ft_learning_rate = 2e-4
        self.ft_max_grad_norm = 0.3
        self.ft_warmup_ratio = 0.03
        self.ft_lr_scheduler_type = "constant"
        self.ft_gradient_checkpointing = True
        self.ft_gradient_checkpointing_use_reentrant = False
        self.ft_optim = "adamw_torch_fused"
        self.ft_save_strategy = "epoch"
        self.ft_logging_steps = 1

        # —— 蒸餾分組 (ds_*) 預設 ——
        # 模型載入（學生）

        self.ds_student_model_id = "google/gemma-3-4b-pt"
        self.ds_processor_id = "google/gemma-3-4b-it"
        self.ds_torch_dtype = "bfloat16"
        self.ds_attn_implementation = "eager"
        self.ds_device_map = "auto"
        self.ds_use_4bit = True
        self.ds_bnb_4bit_use_double_quant = True
        self.ds_bnb_4bit_quant_type = "nf4"
        # SFT
        self.ds_num_train_epochs = 3
        self.ds_per_device_train_batch_size = 1
        self.ds_gradient_accumulation_steps = 4
        self.ds_learning_rate = 1e-4
        self.ds_max_grad_norm = 0.3
        self.ds_warmup_ratio = 0.03
        self.ds_lr_scheduler_type = "constant"
        self.ds_gradient_checkpointing = True
        self.ds_gradient_checkpointing_use_reentrant = False
        self.ds_optim = "adamw_torch_fused"
        self.ds_save_strategy = "epoch"
        self.ds_logging_steps = 1

        # 其他共用
        self.use_wandb = True
        self.wandb_project = "III-2025-golf"
        self.image_scale_percent = 20
        self.image_quality = 50
        self.max_new_tokens = 512
        self.top_p = 0.95
        self.temperature = 0.7
        self.eval_test_count = 5
        self.eval_random_select = False
        self.eval_output_dir = "experiment_result"

        # 蒸餾資料/教師與合併
        self.teacher_model_path = self.model_id
        self.distill_dataset_locate = "distill_teacher_signals.csv"
        self.distill_loss_weight = 0.5
        self.merged_model_path = "model/merged_teacher_model"
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.lora_r = 16
        self.lora_merge_max_shard_size = "5GB"
        self.writer_batch_size = 8

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"更新配置: {key} = {value}")
            else:
                print(f"警告: 未知的配置參數 {key}")
    
    def get_data_path(self):
        if self.data_type == "textonly":
            return f"{self.file_locate}dataset/0513_SFTDataset/text/qa_pairs_sft.json"
        elif self.data_type == "hitdata":
            return f"{self.file_locate}dataset/0513_SFTDataset/hitdata/sft_training_data.json"
        elif self.data_type == "mergedata":
            return f"{self.file_locate}dataset/0513_SFTDataset/mergedata/merged_dataset.json"
        else:
            raise ValueError(f"未知的數據類型: {self.data_type}")

    def get_distill_data_path(self):
        return os.path.join(self.file_locate, "dataset", self.distill_dataset_locate)

    def print_config(self):
        print("=" * 50)
        print("訓練配置:")
        print(f"  實驗名稱: {self.exp_name}")
        print(f"  數據類型: {self.data_type}")
        # 微調
        print("  [Fine-tune]")
        print(f"    model: {self.ft_model_id} | processor: {self.ft_processor_id}")
        print(f"    dtype: {self.ft_torch_dtype} | attn: {self.ft_attn_implementation} | device: {self.ft_device_map} | 4bit: {self.ft_use_4bit}")
        print(f"    epochs: {self.ft_num_train_epochs} | bs: {self.ft_per_device_train_batch_size} | grad_acc: {self.ft_gradient_accumulation_steps}")
        print(f"    lr: {self.ft_learning_rate} | max_grad_norm: {self.ft_max_grad_norm} | warmup: {self.ft_warmup_ratio} | sched: {self.ft_lr_scheduler_type}")
        print(f"    optim: {self.ft_optim} | save: {self.ft_save_strategy} | gckpt: {self.ft_gradient_checkpointing}(reent={self.ft_gradient_checkpointing_use_reentrant}) | log_steps: {self.ft_logging_steps}")
        # 蒸餾
        print("  [Distill]")
        print(f"    student: {self.ds_student_model_id} | processor: {self.ds_processor_id} | teacher: {self.teacher_model_path} | kd_w: {self.distill_loss_weight}")
        print(f"    dtype: {self.ds_torch_dtype} | attn: {self.ds_attn_implementation} | device: {self.ds_device_map} | 4bit: {self.ds_use_4bit}")
        print(f"    epochs: {self.ds_num_train_epochs} | bs: {self.ds_per_device_train_batch_size} | grad_acc: {self.ds_gradient_accumulation_steps}")
        print(f"    lr: {self.ds_learning_rate} | max_grad_norm: {self.ds_max_grad_norm} | warmup: {self.ds_warmup_ratio} | sched: {self.ds_lr_scheduler_type}")
        print(f"    optim: {self.ds_optim} | save: {self.ds_save_strategy} | gckpt: {self.ds_gradient_checkpointing}(reent={self.ds_gradient_checkpointing_use_reentrant}) | log_steps: {self.ds_logging_steps}")
        # 其他
        print(f"  Wandb: {self.use_wandb} | project: {self.wandb_project}")
        print(f"  Eval: count={self.eval_test_count} random={self.eval_random_select} out={self.eval_output_dir}")
        print(f"  Image: scale%={self.image_scale_percent} quality={self.image_quality}")
        print(f"  Merge: shard={self.lora_merge_max_shard_size} out={self.merged_model_path}")
        print("=" * 50)


# 預設配置實例
default_config = TrainingConfig() 