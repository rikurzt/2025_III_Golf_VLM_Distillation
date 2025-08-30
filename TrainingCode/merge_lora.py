#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 PEFT LoRA 適配器與基礎模型合併並儲存為一個獨立的完整模型。
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

def merge_lora_weights(args):
    """
    載入基礎模型和 LoRA 適配器，合併它們，然後儲存結果。
    """
    print(f"正在從 '{args.base_model_id}' 載入基礎模型...")
    
    # 載入基礎模型
    # 使用 bfloat16 以獲得更好的效能和精度，並設定 low_cpu_mem_usage 以處理大型模型
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" # 自動將模型分配到可用設備
    )

    print(f"正在從 '{args.lora_adapter_path}' 載入 LoRA 適配器...")
    # 載入 PeftModel
    peft_model = PeftModel.from_pretrained(model, args.lora_adapter_path)

    print("正在合併 LoRA 權重...")
    # 合併 LoRA 權重與基礎模型
    merged_model = peft_model.merge_and_unload()

    print(f"正在將合併後的模型儲存至 '{args.output_path}'...")
    # 儲存合併後的模型
    # 使用 safe_serialization=True 以安全地儲存模型權重
    merged_model.save_pretrained(
        args.output_path, 
        safe_serialization=True, 
        max_shard_size=args.max_shard_size
    )

    print("正在儲存 processor...")
    # 載入並儲存對應的 processor
    # 這確保了模型和其預處理器可以一起被載入
    try:
        processor = AutoProcessor.from_pretrained(args.lora_adapter_path)
    except Exception as e:
        print(f"從 LoRA 路徑載入 processor 失敗 ({e})，嘗試從基礎模型 '{args.base_model_id}' 載入...")
        processor = AutoProcessor.from_pretrained(args.base_model_id)
        
    processor.save_pretrained(args.output_path)

    print("\nLoRA 合併完成！")
    print(f"合併後的模型已儲存至: {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="合併 PEFT LoRA 適配器與基礎模型")
    parser.add_argument("--base_model_id", type=str, required=True, help="基礎模型的 Hugging Face ID (例如 'google/gemma-3-27b-pt')")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="訓練好的 LoRA 適配器權重所在的路徑")
    parser.add_argument("--output_path", type=str, required=True, help="合併後模型的儲存路徑")
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="儲存模型時的最大分片大小")
    
    args = parser.parse_args()
    merge_lora_weights(args)

if __name__ == "__main__":
    main() 