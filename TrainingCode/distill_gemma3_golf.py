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
    create_distill_data_collator,
    StudentModelWithProjector,
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

        output_hs = self.config.ds_do_hidden_states_loss or self.config.ds_do_image_hidden_states_loss
        output_attn = self.config.ds_do_attentions_loss

        model_kwargs = dict(
            attn_implementation=self.config.ds_attn_implementation,
            torch_dtype=target_dtype,
            device_map=self.config.ds_device_map,
            output_hidden_states=output_hs,
            output_attentions=output_attn,
            do_sample=False,
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
        self.model.config.output_hidden_states = output_hs
        self.model.config.output_attentions = output_attn
        # 確保 vision tower 也會輸出隱藏狀態
        if hasattr(self.model.config, "vision_config") and output_hs:
            self.model.config.vision_config.output_hidden_states = True

        print("學生模型載入完成。")
        return self.model

    def setup_teacher_model(self):
        """
        載入助教模型。此模型將在 `eval` 模式下運行且不計算梯度。
        """
        teacher_model_path = self.config.file_locate + self.config.merged_model_path
        print(f"正在從 {teacher_model_path} 載入助教模型...")
        if not os.path.exists(teacher_model_path):
            raise FileNotFoundError(f"助教模型路徑不存在: {teacher_model_path}。請先合併 LoRA 權重。")
            
        target_dtype = torch.bfloat16 if self.config.ds_torch_dtype == "bfloat16" else torch.float16

        output_hs = self.config.ds_do_hidden_states_loss or self.config.ds_do_image_hidden_states_loss
        output_attn = self.config.ds_do_attentions_loss

        model_kwargs = dict(
            attn_implementation=self.config.ds_attn_implementation,
            torch_dtype=target_dtype,
            device_map=self.config.ds_device_map,
            output_hidden_states=output_hs,
            output_attentions=output_attn,
            do_sample=False,
        )
        if self.config.ds_use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.ds_bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.ds_bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=target_dtype,
                bnb_4bit_quant_storage=target_dtype,
            )
            model_kwargs["quantization_config"] = quantization_config

        teacher_model = AutoModelForImageTextToText.from_pretrained(
            teacher_model_path,
            **model_kwargs,
        )
        
        # 確保 vision tower 也會輸出隱藏狀態
        if hasattr(teacher_model.config, "vision_config") and output_hs:
            teacher_model.config.vision_config.output_hidden_states = True

        teacher_model.eval()
        print("合併助教模型載入完成。")
        return teacher_model

    def setup_training_config(self):
        time = datetime.now().strftime("%Y_%m_%d_%H%M")
        args = SFTConfig(
            output_dir=f"{self.config.file_locate}/model/Distilled{self.config.exp_name}{time}",
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
            save_safetensors=False,
        )
        args.remove_unused_columns = False
        return args, time

    def train(self):
        """執行蒸餾訓練"""
        print("開始蒸餾訓練流程...")
        
        # 1. 載入原始數據集
        print("載入數據集...")
        dataset = self.load_dataset()
        print(f"載入 {len(dataset)} 筆資料用於蒸餾。")
        
        # 2. 設置學生模型、教師模型和處理器
        self.setup_student_model()
        teacher_model = self.setup_teacher_model()
        self.processor = AutoProcessor.from_pretrained(self.config.ds_processor_id)

        student_hidden_size = self.model.config.text_config.hidden_size
        teacher_hidden_size = teacher_model.config.text_config.hidden_size

        # 建立 projector 維度資訊（文字與影像）
        student_image_hidden_size = self.model.config.vision_config.hidden_size if hasattr(self.model.config, "vision_config") else None
        teacher_image_hidden_size = teacher_model.config.vision_config.hidden_size if hasattr(teacher_model.config, "vision_config") else None

        # 建立 projector：
        # - t2s: 教師文字 hidden -> 學生文字 hidden
        # - img_t2s: 教師影像 hidden -> 學生影像 hidden（僅一層 Linear）
        if student_hidden_size != teacher_hidden_size or (
            student_image_hidden_size is not None and teacher_image_hidden_size is not None and student_image_hidden_size != teacher_image_hidden_size
        ):
            print(f"學生/教師文字隱藏維度: {student_hidden_size} / {teacher_hidden_size}")
            if student_image_hidden_size is not None and teacher_image_hidden_size is not None:
                print(f"學生/教師影像隱藏維度: {student_image_hidden_size} / {teacher_image_hidden_size}")
            print("正在建立線性投影層（文字與影像）...")
            projector = torch.nn.ModuleDict({
                "t2s": torch.nn.Linear(teacher_hidden_size, student_hidden_size) if student_hidden_size != teacher_hidden_size else torch.nn.Identity(),
                "img_t2s": (
                    torch.nn.Linear(teacher_image_hidden_size, student_image_hidden_size)
                    if (student_image_hidden_size is not None and teacher_image_hidden_size is not None and student_image_hidden_size != teacher_image_hidden_size)
                    else torch.nn.Identity()
                ),
            })
        else:
            print(f"學生與教師隱藏維度相同 (text={student_hidden_size}, image={student_image_hidden_size})，使用恒等映射 projector。")
            projector = torch.nn.ModuleDict({
                "t2s": torch.nn.Identity(),
                "img_t2s": torch.nn.Identity(),
            })
        # 將 projector 移動到與學生模型相同的裝置與 dtype
        student_param = next(self.model.parameters())
        projector.to(student_param.device, dtype=student_param.dtype)
        self.model = StudentModelWithProjector(self.model, projector)
        print("已將學生模型與 projector 打包完成。")
        
        # 3. 設置訓練配置
        args, time = self.setup_training_config()
        
        # 4. 創建蒸餾資料整理器
        distill_collate_fn = create_distill_data_collator(
            self.processor
        )
        
        # 5. 創建蒸餾訓練器
        self.trainer = DistillSTFTrainer(
            model=self.model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=dataset,
            processing_class=self.processor,
            data_collator=distill_collate_fn,
            distill_rate=self.config.distill_rate,
            do_hidden_states_loss=self.config.ds_do_hidden_states_loss,
            do_attentions_loss=self.config.ds_do_attentions_loss,
            do_image_hidden_states_loss=self.config.ds_do_image_hidden_states_loss,
            hidden_states_last_only=self.config.ds_hidden_states_last_only,
        )
        
        # 初始化wandb
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, name=f"distill-{self.config.exp_name}{time}", config=vars(self.config))

        # 開始訓練
        print("開始模型蒸餾訓練...")
        self.trainer.train()
        
        # 保存模型
        print("保存模型...")
        output_dir = args.output_dir
        base_model = self.model.model if isinstance(self.model, StudentModelWithProjector) else self.model
        base_model.save_pretrained(output_dir, safe_serialization=False)
        self.processor.save_pretrained(output_dir)
        
        # 清理
        del self.model
        del self.trainer
        del teacher_model
        torch.cuda.empty_cache()
        
        print("蒸餾訓練完成！")
        return args.output_dir

if __name__ == "__main__":
    config = TrainingConfig()
    distill_trainer = GolfDistillationTrainer(config)
    output_dir = distill_trainer.train()
    print(f"蒸餾後的學生模型已保存至: {output_dir}") 