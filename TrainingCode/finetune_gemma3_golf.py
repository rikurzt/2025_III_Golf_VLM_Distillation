#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma3-4b 高爾夫球訓練數據微調腳本
從 FineTune_0513Golf_Data_Gemma3-4b.ipynb 整理而來
"""

import os
import json
import torch
import wandb
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from config import TrainingConfig

class GolfDatasetTrainer:
    def __init__(self, config: TrainingConfig):
        """初始化訓練器"""
        self.config = config
        self.model = None
        self.processor = None
        self.trainer = None
        self.dataset = None
    
    def _get_data_path(self):
        """根據數據類型獲取數據路徑"""
        return self.config.get_data_path()

    def _val(self, ft_value, common_value):
        return ft_value if ft_value is not None else common_value
    
    def base64_to_pil(self, base64_str: str) -> Image.Image:
        """將 base64 字串轉為 PIL Image"""
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))
    
    def compress_image(self, img: Image.Image, scale_percent: int = 20, quality: int = 50) -> Image.Image:
        """
        根據原圖大小以百分比縮小圖片。
        - scale_percent: 縮小比例（例如 50 表示縮成 50%）
        - quality: JPEG 壓縮品質（1~100）
        回傳縮小並壓縮後的 PIL.Image 物件。
        """
        img = img.convert("RGB")  # 確保轉成 RGB 模式
        width, height = img.size
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # 壓縮品質模擬：寫入 buffer 再重新讀入
        buffer = BytesIO()
        resized_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).copy()  # ← 強制 load 到記憶體

    def convert_images_in_dataset(self, dataset):
        """
        將 dataset 中所有 base64 圖片轉換為 PIL.Image，保留原資料結構
        """
        print("正在轉換數據集中的圖片...")
        for entry in dataset:
            for message in entry.get("messages", []):
                content_list = message.get("content", [])
                
                # 處理 content 可能是字串的情況
                if isinstance(content_list, str):
                    continue  # 跳過純文字內容
                
                # 確保 content_list 是列表
                if not isinstance(content_list, list):
                    content_list = [content_list]
                
                for content in content_list:
                    # 確保 content 是字典
                    if not isinstance(content, dict):
                        continue
                        
                    if content.get("type") == "image" and isinstance(content.get("image"), str):
                        try:
                            img = self.base64_to_pil(content["image"])
                            img = self.compress_image(img, scale_percent=self.config.image_scale_percent, quality=self.config.image_quality)
                            content["image"] = img
                        except Exception as e:
                            print(f"圖片轉換錯誤: {e}")
                            content["image"] = None  # 或保留原 base64

        return dataset  # 可省略，dataset 是 in-place 修改
    
    def load_dataset(self):
        """載入數據集"""
        data_path = self._get_data_path()
        print(f"載入數據集: {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        # 如果是 hitdata 或 mergedata 類型，需要轉換圖片
        if self.config.data_type in ["hitdata", "mergedata"]:
            self.dataset = self.convert_images_in_dataset(self.dataset)
        
        print(f"數據集載入完成，共 {len(self.dataset)} 筆資料")
        return self.dataset
    
    def process_vision_info(self, messages: list[dict]) -> list[Image.Image]:
        """處理訊息中的視覺資訊"""
        image_inputs = []
        # Iterate through each conversation
        
        for msg in messages:
            # Get content (ensure it's a list)
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            # Check each content element for images
            for element in content:
                if isinstance(element, dict) and (
                    "image" in element or element.get("type") == "image"
                ):
                    # Get the image and convert to RGB
                    if "image" in element:
                        image = element["image"]
                    else:
                        image = element
                    image_inputs.append(image.convert("RGB"))
        return image_inputs

    def collate_fn(self, examples):
        """資料整理函數"""
        """
        1. 檢測每個樣本是否包含圖片
        2. 分別處理有圖片和無圖片的樣本
        3. 只有在有圖片時才處理圖片相關的token
        """
        texts = []
        images = []
        
        for example in examples:
            # 提取圖片
            image_inputs = self.process_vision_info(example["messages"])
            
            # 生成對話文本
            text = self.processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            
            # 處理圖片：如果沒有圖片則添加None
            if len(image_inputs) == 0:
                images.append(None)
            else:
                images.append(image_inputs)

        # 檢查是否有任何圖片
        has_images = any(img is not None for img in images)
        
        if has_images:
            # 處理圖文對
            processed_images = []
            for img_list in images:
                if img_list is None:
                    continue  # 跳過沒有圖片的樣本
                else:
                    processed_images.extend(img_list)
            
            # Tokenize the texts and process the images
            batch = self.processor(
                text=texts, 
                images=processed_images if processed_images else None, 
                return_tensors="pt", 
                padding=True
            )
        else:
            # 純文字處理
            batch = self.processor(text=texts, images=None, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()
        
        # Mask tokens for not being used in the loss computation
        # 只有當有圖片時才處理圖片token
        if has_images and any(img is not None for img in images):
            try:
                # 查找圖片token ID
                if "boi_token" in self.processor.tokenizer.special_tokens_map:
                    image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                        self.processor.tokenizer.special_tokens_map["boi_token"]
                    )
                    labels[labels == image_token_id] = -100
                
                # 其他圖片相關的token
                labels[labels == 262144] = -100
                
            except (KeyError, AttributeError):
                # 如果沒有圖片相關的特殊token，跳過
                pass
        
        # Mask image tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch
    
    def setup_model(self):
        """設置模型和處理器"""
        print("設置模型...")
        attn_impl = self._val(self.config.ft_attn_implementation, self.config.ft_attn_implementation)
        torch_dtype_cfg = self._val(self.config.ft_torch_dtype, self.config.ft_torch_dtype)
        print(f"使用配置: attn_implementation={attn_impl}, torch_dtype={torch_dtype_cfg}")
        
        # Check if GPU benefits from bfloat16
        if torch.cuda.get_device_capability()[0] < 8:
            raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

        # 從配置中讀取 torch_dtype
        if torch_dtype_cfg == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype_cfg == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16  # 默認值

        # Define model init arguments (從配置中讀取)
        model_kwargs = dict(
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            device_map=self._val(self.config.ft_device_map, self.config.ft_device_map),
        )

        # BitsAndBytesConfig int-4 config (可由 config 控制)
        use_4bit = self._val(self.config.ft_use_4bit, self.config.ft_use_4bit)
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self._val(self.config.ft_bnb_4bit_use_double_quant, self.config.ft_bnb_4bit_use_double_quant),
                bnb_4bit_quant_type=self._val(self.config.ft_bnb_4bit_quant_type, self.config.ft_bnb_4bit_quant_type),
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_storage=torch_dtype,
            )

        # Load model and tokenizer (從配置讀取)
        model_id = self._val(self.config.ft_model_id, self.config.model_id)
        processor_id = self._val(self.config.ft_processor_id, self.config.processor_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(processor_id, use_fast=True)
        
        print("模型設置完成")
        return self.model, self.processor
    
    def setup_training_config(self):
        """設置訓練配置"""
        time = datetime.now().strftime("%Y_%m_%d_%H%M")
        
        args = SFTConfig(
            output_dir=f"{self.config.file_locate}/model/{self.config.exp_name}{time}",
            num_train_epochs=self._val(self.config.ft_num_train_epochs, self.config.ft_num_train_epochs),
            per_device_train_batch_size=self._val(self.config.ft_per_device_train_batch_size, self.config.ft_per_device_train_batch_size),
            gradient_accumulation_steps=self._val(self.config.ft_gradient_accumulation_steps, self.config.ft_gradient_accumulation_steps),
            gradient_checkpointing=self._val(self.config.ft_gradient_checkpointing, self.config.ft_gradient_checkpointing),
            optim=self._val(self.config.ft_optim, self.config.ft_optim),
            logging_steps=self._val(self.config.ft_logging_steps, self.config.ft_logging_steps),
            logging_dir="logs",
            save_strategy=self._val(self.config.ft_save_strategy, self.config.ft_save_strategy),
            learning_rate=self._val(self.config.ft_learning_rate, self.config.ft_learning_rate),
            bf16=(self._val(self.config.ft_torch_dtype, self.config.ft_torch_dtype) == "bfloat16"),
            max_grad_norm=self._val(self.config.ft_max_grad_norm, self.config.ft_max_grad_norm),
            warmup_ratio=self._val(self.config.ft_warmup_ratio, self.config.ft_warmup_ratio),
            lr_scheduler_type=self._val(self.config.ft_lr_scheduler_type, self.config.ft_lr_scheduler_type),
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else "none",
            gradient_checkpointing_kwargs={
                "use_reentrant": self._val(self.config.ft_gradient_checkpointing_use_reentrant, self.config.ft_gradient_checkpointing_use_reentrant)
            },
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            run_name=self.config.exp_name + time,
        )
        args.remove_unused_columns = False  # important for collator
        
        return args, time
    
    def setup_peft_config(self):
        """設置PEFT配置"""
        peft_config = LoraConfig(
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            r=self.config.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=[
                "lm_head",
                "embed_tokens",
            ],
        )
        return peft_config
    
    def train(self):
        """執行訓練"""
        print("開始訓練流程...")
        
        # 載入數據集
        self.load_dataset()
        
        # 設置模型
        self.setup_model()
        
        # 設置訓練配置
        args, time = self.setup_training_config()
        
        # 設置PEFT配置
        peft_config = self.setup_peft_config()
        
        # 創建訓練器
        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset,
            peft_config=peft_config,
            processing_class=self.processor,
            data_collator=self.collate_fn,
        )
        
        # 初始化wandb (如果啟用的話)
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, name=self.config.exp_name + time)
        
        # 開始訓練
        print("開始模型訓練...")
        self.trainer.train()
        
        # 保存模型
        print("保存模型...")
        self.trainer.save_model()
        
        # 清理記憶體
        print("清理記憶體...")
        del self.model
        del self.trainer
        torch.cuda.empty_cache()
        
        print("訓練完成！")
        return args.output_dir


def main():
    """主函數"""
    # 創建訓練器實例
    trainer = GolfDatasetTrainer(TrainingConfig())
    
    # 執行訓練
    output_dir = trainer.train()
    
    print(f"模型已保存至: {output_dir}")


if __name__ == "__main__":
    main() 