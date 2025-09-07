#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從已訓練好的教師模型生成蒸餾數據（teacher signals）。
"""

import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
import gc
import gzip

from config import TrainingConfig
# 匯入微調腳本中的數據處理邏輯
from finetune_gemma3_golf import GolfDatasetTrainer
from distillation import to_tensor_list

def pin_and_to_device(batch, device):
    """將 CPU 張量 pin 住並以 non_blocking 搬移到指定裝置。"""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.device.type == "cpu":
                try:
                    tensor = tensor.pin_memory()
                except Exception:
                    pass
            moved[key] = tensor.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved

def generate_distill_data(args):
    """主函數：生成並保存蒸餾數據"""

    # 1. 載入數據集
    # 我們借用 GolfDatasetTrainer 來處理數據集載入和圖片轉換
    # 這裡我們只需要它的數據處理能力，而不是完整的訓練流程
    data_handler = GolfDatasetTrainer(TrainingConfig())
    # 手動設定配置以匹配數據類型
    data_handler.config.data_type = args.data_type
    data_handler.config.file_locate = args.file_locate
    dataset = data_handler.load_dataset()

    # 2. 載入助教模型和處理器
    print(f"正在從 {args.teacher_model_path} 載入教師模型...")
    # 這裡我們假設教師模型已經合併 LoRA 權重並且是完整模型
    # 蒸餾時通常使用 bfloat16 以獲得高質量的輸出
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # or flash_attention_2 if available
        output_hidden_states = True,
        output_attentions = True
    )
    processor = AutoProcessor.from_pretrained(args.processor_path)

    # 確保模型會輸出所需的中間特徵
    model.config.output_hidden_states = True
    model.config.output_attentions = True
    if hasattr(model.config, "vision_config"):
        model.config.vision_config.output_hidden_states = True
    model.eval()

    # 準備輸出檔案（逐筆寫入避免累積記憶體）
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"蒸餾訊號將逐筆儲存於目錄：{output_dir}")

    # 3. 生成 Teacher Signals
    print("正在逐筆生成並保存 teacher signals 到 .pt 檔案...")
    count = 0
    for idx, sample in enumerate(tqdm(dataset, desc="處理樣本中")):
        
        text_prompt = processor.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        ).strip()
        
        image_inputs = data_handler.process_vision_info(sample["messages"])
        
        # 處理沒有圖片的樣本
        if not image_inputs:
            inputs_cpu = processor(text=[text_prompt], images=None, return_tensors="pt", padding=False)
        else:
            inputs_cpu = processor(text=[text_prompt], images=image_inputs, return_tensors="pt", padding=False)
        # 將 CPU 張量 pin 並非同步搬移到 GPU
        inputs = pin_and_to_device(inputs_cpu.data if hasattr(inputs_cpu, "data") else dict(inputs_cpu), model.device)

        with torch.inference_mode():
            out = model(**inputs)

        # 將輸出立即搬到 CPU，避免占用 VRAM
        hs_list_cpu = [t.detach().to("cpu") for t in to_tensor_list(getattr(out, "hidden_states", None))]
        attn_list_cpu = [t.detach().to("cpu") for t in to_tensor_list(getattr(out, "attentions", None))]
        img_hs_list_cpu = [t.detach().to("cpu") for t in to_tensor_list(getattr(out, "image_hidden_states", None))]

        # 為當前樣本建立子目錄
        sample_output_dir = os.path.join(output_dir, str(idx))
        os.makedirs(sample_output_dir, exist_ok=True)

        # 將每個張量列表儲存到獨立的 .pt 檔案
        if hs_list_cpu:
            with gzip.open(os.path.join(sample_output_dir, "teacher_hidden_states.pt.gz"), "wb") as f:
                torch.save(hs_list_cpu, f)
        if attn_list_cpu:
            with gzip.open(os.path.join(sample_output_dir, "teacher_attentions.pt.gz"), "wb") as f:
                torch.save(attn_list_cpu, f)
        if img_hs_list_cpu:
            with gzip.open(os.path.join(sample_output_dir, "teacher_image_hidden_states.pt.gz"), "wb") as f:
                torch.save(img_hs_list_cpu, f)

        # 釋放 GPU/CPU 資源
        del out, inputs, inputs_cpu, hs_list_cpu, attn_list_cpu, img_hs_list_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        count += 1

    # 4. 完成訊息
    print(f"已保存 {count} 筆資料到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="為模型蒸餾生成 Teacher Signals")
    parser.add_argument("--teacher_model_path", type=str, required=True, help="已訓練好的教師模型路徑 (通常是合併LoRA後的模型)")
    parser.add_argument("--processor_path", type=str, default="google/gemma-3-4b-it", help="模型對應的 processor 路徑")
    parser.add_argument("--data_type", type=str, choices=["textonly", "hitdata", "mergedata"], required=True, help="要處理的數據集類型")
    parser.add_argument("--file_locate", type=str, default="./", help="專案根目錄路徑")
    parser.add_argument("--output_dir", type=str, default="dataset/distill_teacher_signals", help="輸出 pt 檔案的目錄")

    args = parser.parse_args()
    generate_distill_data(args)

if __name__ == "__main__":
    main() 