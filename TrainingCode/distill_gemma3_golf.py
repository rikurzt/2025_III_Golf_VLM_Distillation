#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma3-4b 高爾夫球數據蒸餾訓練腳本
"""

import os
import copy
import torch
import wandb
from datetime import datetime

from transformers import AutoProcessor, AutoModelForImageTextToText, Gemma3ForConditionalGeneration,BitsAndBytesConfig
from trl import SFTConfig
from peft import LoraConfig

# 匯入本地模組
from config import TrainingConfig
from finetune_gemma3_golf import GolfDatasetTrainer
from distillation import (
    DistillSTFTrainer, 
    CSVTeacherSignalsDataset, 
    create_distill_data_collator,
)


class GolfDistillationTrainer(GolfDatasetTrainer):
    def __init__(self, config: TrainingConfig):
        """初始化蒸餾訓練器"""
        super().__init__(config)

    def setup_student_model(self):
        """
        載入學生模型。
        """
        print(f"正在從 {self.config.ds_student_model_id} 載入學生模型...")
        target_dtype = torch.bfloat16 if self.config.ds_torch_dtype == "bfloat16" else torch.float16

        model_kwargs = dict(
            attn_implementation=self.config.ds_attn_implementation,
            torch_dtype=target_dtype,
            device_map=self.config.ds_device_map,
            output_hidden_states=True,
            output_attentions=True,
        )
        # BitsAndBytesConfig int-4 量化配置 (由 config 控制)
        if self.config.ds_use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.ds_bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.ds_bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=target_dtype,
                bnb_4bit_quant_storage=target_dtype,
            )
            #model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.ds_student_model_id,
            **model_kwargs,
        )
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        # 確保 vision tower 也會輸出隱藏狀態
        if hasattr(self.model.config, "vision_config"):
            self.model.config.vision_config.output_hidden_states = True

        print("學生模型載入完成。")
        return self.model

    def setup_training_config(self):
        time = datetime.now().strftime("%Y_%m_%d_%H%M")
        args = SFTConfig(
            output_dir=f"{self.config.file_locate}/model/{self.config.exp_name}{time}",
            num_train_epochs=self.config.ds_num_train_epochs,
            per_device_train_batch_size=self.config.ds_per_device_train_batch_size,
            gradient_accumulation_steps=self.config.ds_gradient_accumulation_steps,
            gradient_checkpointing=self.config.ds_gradient_checkpointing,
            optim=self.config.ds_optim,
            logging_steps=self.config.ds_logging_steps,
            logging_dir="logs",
            save_strategy=self.config.ds_save_strategy,
            learning_rate=self.config.ds_learning_rate,
            bf16=(self.config.ds_torch_dtype == "bfloat16"),
            max_grad_norm=self.config.ds_max_grad_norm,
            warmup_ratio=self.config.ds_warmup_ratio,
            lr_scheduler_type=self.config.ds_lr_scheduler_type,
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else "none",
            gradient_checkpointing_kwargs={
                "use_reentrant": self.config.ds_gradient_checkpointing_use_reentrant
            },
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            run_name=self.config.exp_name + time,
        )
        args.remove_unused_columns = False
        return args, time

    def train(self):
        """執行蒸餾訓練"""
        print("開始蒸餾訓練流程...")
        
        # 1. 載入原始數據集 (用於結構) 和蒸餾數據集
        print("載入蒸餾數據集...")
        raw_dataset_handler = GolfDatasetTrainer(self.config)
        raw_dataset = raw_dataset_handler.load_dataset()
        
        distill_data_path = self.config.get_distill_data_path()
        distill_dataset = CSVTeacherSignalsDataset(distill_data_path, raw_dataset)
        print(f"從 {distill_data_path} 載入 {len(distill_dataset)} 筆蒻餾資料。")

        
        # 2. 設置學生模型和處理器
        self.setup_student_model()
        self.processor = AutoProcessor.from_pretrained(self.config.ds_processor_id)

        # 3. 設置訓練配置
        args, time = self.setup_training_config()
        
        # 5. 創建蒸餾資料整理器
        distill_collate_fn = create_distill_data_collator(
            self.processor
        )
        
        # 6. 創建蒸餾訓練器（帶蒸餾權重）
        self.trainer = DistillSTFTrainer(
            model=self.model,
            args=args,
            train_dataset=distill_dataset,
            processing_class=self.processor,
            data_collator=distill_collate_fn,
            distill_weight=self.config.distill_loss_weight,
        )
        
        # 初始化wandb
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, name=f"distill-{self.config.exp_name}{time}", config=vars(self.config))

        # 開始訓練
        print("開始模型蒸餾訓練...")
        self.trainer.train()
        
        # 保存模型
        print("保存模型...")
        self.trainer.save_model()
        
        # 清理
        del self.model
        del self.trainer
        torch.cuda.empty_cache()
        
        print("蒸餾訓練完成！")
        return args.output_dir

if __name__ == "__main__":
    config = TrainingConfig()
    distill_trainer = GolfDistillationTrainer(config)
    output_dir = distill_trainer.train()
    print(f"蒸餾後的學生模型已保存至: {output_dir}") 