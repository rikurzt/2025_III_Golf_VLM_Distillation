#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高爾夫球 VLM 訓練執行腳本
使用範例
"""

import argparse
from train_gemma3_golf import GolfDatasetTrainer
from config import TrainingConfig


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="高爾夫球 VLM 訓練執行腳本")
    
    # 基本參數
    parser.add_argument("--exp_name", type=str, default="gemma3-4b-sft-textonly-", 
                       help="實驗名稱")
    parser.add_argument("--data_type", type=str, choices=["textonly", "hitdata"], 
                       default="textonly", help="數據類型")
    parser.add_argument("--file_locate", type=str, default="/tmp/pycharm_project_979/",
                       help="文件根目錄路徑")
    
    # 訓練參數
    parser.add_argument("--num_epochs", type=int, default=5, help="訓練週期數")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="學習率")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    # 其他參數
    parser.add_argument("--wandb_project", type=str, default="III-2025-golf", 
                       help="Wandb 專案名稱")
    parser.add_argument("--no_wandb", action="store_true", help="不使用 wandb")
    
    args = parser.parse_args()
    
    # 創建配置
    config = TrainingConfig()
    

    
    # 打印配置
    config.print_config()
    
    # 創建訓練器
    trainer = GolfDatasetTrainer()
    trainer.config = config
    
    try:
        # 執行訓練
        output_dir = trainer.train()
        print(f"\n訓練完成，模型已保存至: {output_dir}")
        
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    main() 