#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高爾夫球 VLM 訓練執行腳本
使用範例
"""

import argparse
from train_gemma3_golf import GolfDatasetTrainer
from evaluate import GolfModelEvaluator
from config import TrainingConfig


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="高爾夫球 VLM 訓練執行腳本")
    
    # 只保留流程控制參數
    parser.add_argument("--skip_evaluation", action="store_true", help="跳過訓練後的評估階段")
    
    args = parser.parse_args()

    # 創建配置（使用 config.py 中的預設參數）
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
        
        # 執行評估 (如果未跳過)
        if not args.skip_evaluation:
            print("\n" + "="*60)
            print("開始自動評估階段...")
            print("="*60)
            
            # 創建評估器
            evaluator = GolfModelEvaluator(config=config)
            
            try:
                # 載入剛訓練完的模型
                evaluator.load_model(output_dir)
                
                # 執行評估（使用 config 中的參數）
                results = evaluator.evaluate_model(
                    test_count=config.eval_test_count,
                    output_dir=config.eval_output_dir,
                    random_select=config.eval_random_select,
                    save_csv=True
                )
                
                print(f"\n評估完成，共測試 {len(results)} 筆資料")
                
                # 統計成功率
                successful_tests = sum(1 for r in results if r['evaluation_status'] == 'SUCCESS')
                success_rate = successful_tests / len(results) * 100 if results else 0
                print(f"評估成功率: {success_rate:.1f}% ({successful_tests}/{len(results)})")
                
            except Exception as e:
                print(f"\n評估過程中發生錯誤: {e}")
                print("訓練已完成，但評估失敗。您可以稍後手動執行評估。")
            finally:
                evaluator.cleanup()
        else:
            print("\n已跳過自動評估階段")
        
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    main() 