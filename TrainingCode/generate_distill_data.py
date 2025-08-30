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

from config import TrainingConfig
# 匯入微調腳本中的數據處理邏輯
from finetune_gemma3_golf import GolfDatasetTrainer
from distillation import serialize_tensor_list, to_tensor_list

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
    output_path = os.path.join(output_dir, args.output_filename)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    # 同步寫檔：初始化表頭寫入旗標
    header_written = False

    # 3. 生成 Teacher Signals
    print("正在逐筆生成並保存 teacher signals 到 CSV...")
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

        # 同步：立刻序列化並寫入一列，保證順序與 idx 一致
        row = {
            "id": idx,
            "teacher_hidden_states_b64": serialize_tensor_list(hs_list_cpu),
            "teacher_attentions_b64": serialize_tensor_list(attn_list_cpu),
            "teacher_image_hidden_states_b64": serialize_tensor_list(img_hs_list_cpu),
        }
        row_df = pd.DataFrame([row])
        row_df.to_csv(output_path, mode="a", header=(not header_written), index=False)
        header_written = True

        # 釋放 GPU/CPU 資源
        del out, inputs, inputs_cpu, hs_list_cpu, attn_list_cpu, img_hs_list_cpu, row_df, row
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        count += 1

    # 4. 完成訊息
    print(f"已保存 {count} 筆資料到 {output_path}")

def main():
    parser = argparse.ArgumentParser(description="為模型蒸餾生成 Teacher Signals")
    parser.add_argument("--teacher_model_path", type=str, required=True, help="已訓練好的教師模型路徑 (通常是合併LoRA後的模型)")
    parser.add_argument("--processor_path", type=str, default="google/gemma-3-4b-it", help="模型對應的 processor 路徑")
    parser.add_argument("--data_type", type=str, choices=["textonly", "hitdata", "mergedata"], required=True, help="要處理的數據集類型")
    parser.add_argument("--file_locate", type=str, default="./", help="專案根目錄路徑")
    parser.add_argument("--output_dir", type=str, default="dataset", help="輸出 CSV 檔案的目錄")
    parser.add_argument("--output_filename", type=str, default="distill_teacher_signals.csv", help="輸出 CSV 檔案的名稱")
    # 兼容舊參數，但已不使用（保持為 no-op）
    parser.add_argument("--writer_batch_size", type=int, default=1, help="已停用：同步模式不使用批次寫入")
    
    args = parser.parse_args()
    generate_distill_data(args)

if __name__ == "__main__":
    main() 