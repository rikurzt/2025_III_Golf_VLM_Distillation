#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高爾夫球 VLM 訓練、評估與蒸餾的整合執行腳本
"""

import argparse
from config import TrainingConfig

def run_finetune(config: TrainingConfig):
    """執行微調流程"""
    from finetune_gemma3_golf import GolfDatasetTrainer
    print("--- 開始微調訓練 ---")
    trainer = GolfDatasetTrainer(config)
    output_dir = trainer.train()
    print(f"--- 微調完成，模型已保存至: {output_dir} ---")
    return output_dir

def run_evaluation(config: TrainingConfig, model_path: str):
    """執行評估流程"""
    from evaluate import GolfModelEvaluator
    print(f"\n--- 開始評估模型: {model_path} ---")
    evaluator = GolfModelEvaluator(config=config)
    try:
        evaluator.load_model(model_path)
        results = evaluator.evaluate_model(
            test_count=config.eval_test_count,
            output_dir=config.eval_output_dir,
            random_select=config.eval_random_select,
            save_csv=True
        )
        successful_tests = sum(1 for r in results if r['evaluation_status'] == 'SUCCESS')
        success_rate = successful_tests / len(results) * 100 if results else 0
        print(f"評估成功率: {success_rate:.1f}% ({successful_tests}/{len(results)})")
    finally:
        evaluator.cleanup()
    print("--- 評估完成 ---")

def run_generate_distill_data(config: TrainingConfig):
    """執行生成蒸餾數據的流程"""
    from generate_distill_data import generate_distill_data
    print("--- 開始生成蒸餾數據 ---")
    # 將 config 物件轉換為 argparse-like 物件
    args = argparse.Namespace(
        teacher_model_path=config.teacher_model_path,
        processor_path=config.processor_id,
        data_type=config.data_type,
        file_locate=config.file_locate,
        output_dir=config.distill_dataset_locate,
        writer_batch_size=getattr(config, "writer_batch_size", 1),
        writer_queue_maxsize=getattr(config, "writer_queue_maxsize", 2)
    )
    generate_distill_data(args)
    print(f"--- 蒸餾數據生成完成，已保存至: {config.distill_dataset_locate} ---")

def run_distill(config: TrainingConfig):
    """執行蒸餾訓練流程"""
    from distill_gemma3_golf import GolfDistillationTrainer
    print("--- 開始蒸餾訓練 ---")
    distill_trainer = GolfDistillationTrainer(config)
    output_dir = distill_trainer.train()
    print(f"--- 蒸餾訓練完成，學生模型已保存至: {output_dir} ---")
    return output_dir

def run_merge_lora(config: TrainingConfig, lora_adapter_path: str):
    """執行合併 LoRA 權重的流程"""
    from merge_lora import merge_lora_weights
    print("--- 開始合併 LoRA 權重 ---")
    # 教師模型通常是微調時使用的基礎模型
    base_model_id = config.model_id
    output_path = config.merged_model_path

    args = argparse.Namespace(
        base_model_id=base_model_id,
        lora_adapter_path=lora_adapter_path,
        output_path=output_path,
        max_shard_size=config.lora_merge_max_shard_size
    )
    merge_lora_weights(args)
    print(f"--- LoRA 合併完成，模型已儲存至: {output_path} ---")
    return output_path

def main():
    """主函數，處理命令列參數和指令"""
    parser = argparse.ArgumentParser(description="高爾夫球 VLM 專案執行腳本")
    subparsers = parser.add_subparsers(dest="command", required=False, help="可執行的指令。若不指定，則執行完整流程。")


    parent_parser = argparse.ArgumentParser(add_help=False)
    for key, value in TrainingConfig().__dict__.items():
        arg_type = type(value) if value is not None else str
        if isinstance(value, bool):
            # 對於 bool，提供 --flag / --no-flag
            parent_parser.add_argument(f"--{key}", dest=key, action='store_true', help=f"設定 {key} (預設: {value})")
            parent_parser.add_argument(f"--no-{key}", dest=key, action='store_false')
            parent_parser.set_defaults(**{key: value})
        else:
            parent_parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"設定 {key} (預設: {value})")

    # --- 'finetune' 指令 ---
    parser_finetune = subparsers.add_parser("finetune", help="執行模型微調訓練", parents=[parent_parser])
    parser_finetune.add_argument("--skip_evaluation", action="store_true", help="訓練後跳過評估")

    # --- 'evaluate' 指令 ---
    parser_evaluate = subparsers.add_parser("evaluate", help="獨立執行模型評估", parents=[parent_parser])
    parser_evaluate.add_argument("--model_path", type=str, required=True, help="要評估的模型路徑")

    # --- 'merge-lora' 指令 ---
    parser_merge = subparsers.add_parser("merge-lora", help="合併 LoRA 適配器與基礎模型", parents=[parent_parser])
    parser_merge.add_argument("--lora_adapter_path", type=str, required=True, help="訓練好的 LoRA 適配器權重所在的路徑")
    parser_merge.add_argument("--base_model_id", type=str, default="google/gemma-3-4b-pt", help="基礎模型的 Hugging Face ID (例如 'google/gemma-3-27b-pt')")
    # --- 'generate-distill-data' 指令 ---
    parser_generate = subparsers.add_parser("generate-distill-data", help="為蒸餾生成教師訊號", parents=[parent_parser])

    # --- 'distill' 指令 ---
    parser_distill = subparsers.add_parser("distill", help="執行蒸餾訓練", parents=[parent_parser])
    parser_distill.add_argument("--skip_evaluation", action="store_true", help="訓練後跳過評估")

    args = parser.parse_args()

    # 創建配置實例並用命令列參數更新
    config = TrainingConfig()
    config.update_config(**vars(args))
    config.print_config()

    # 根據指令執行對應的流程
    try:
        if args.command is None:
            # 1. 微調
            finetuned_lora_path = run_finetune(config)
            print(f"\n--- 微調階段完成 ---")

            # 2. 合併 LoRA
            merged_model_path = run_merge_lora(config, finetuned_lora_path)
            print(f"\n--- LoRA 合併階段完成 ---")

            # 3. 更新設定，將合併後的模型作為教師模型
            print(f"--- 更新設定: 教師模型路徑設定為 {merged_model_path} ---")
            config.teacher_model_path = merged_model_path

            # 4. 生成蒸餾資料
            run_generate_distill_data(config)
            print(f"\n--- 蒸餾資料生成階段完成 ---")

            # 5. 執行蒸餾
            distilled_lora_path = run_distill(config)
            print(f"\n--- 蒸餾階段完成 ---")
            
            # 6. 評估蒸餾後的模型
            print("\n--- 開始評估蒸餾後的學生模型 ---")
            run_evaluation(config, distilled_lora_path)
            
            print("\n--- 完整流程執行完畢 ---")

        elif args.command == "finetune":
            output_dir = run_finetune(config)
            if not args.skip_evaluation:
                run_evaluation(config, output_dir)
        
        elif args.command == "evaluate":
            run_evaluation(config, args.model_path)
            
        elif args.command == "merge-lora":
            # 執行時需要 lora_adapter_path，但如果從 config 初始化，可能不會有
            # args 中會有，所以直接用
            run_merge_lora(config, args.lora_adapter_path)

        elif args.command == "generate-distill-data":
            run_generate_distill_data(config)
            
        elif args.command == "distill":
            output_dir = run_distill(config)
            if not args.skip_evaluation:
                run_evaluation(config, output_dir)

    except Exception as e:
        print(f"\n執行過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main() 