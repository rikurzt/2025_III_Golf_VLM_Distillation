#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高爾夫球 VLM 模型評估腳本
從 FineTune_0513Golf_Data_Gemma3-4b.ipynb 的 Evaluate 階段整理而來
"""

import os
import json
import torch
import random
import pandas as pd
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import TrainingConfig


class GolfModelEvaluator:
    def __init__(self, model_path=None, config=None):
        """初始化評估器
        
        Args:
            model_path: 模型路徑，如果為None則需要後續設定
            config: 配置對象，如果為None則使用默認配置
        """
        self.config = config if config else TrainingConfig()
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.dataset = None
        
    def load_model(self, model_path):
        """載入已訓練的模型
        
        Args:
            model_path: 模型路徑
        """
        print(f"載入模型從: {model_path}")
        
        target_dtype = torch.bfloat16 if self.config.torch_dtype == "bfloat16" else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map=self.config.device_map,
            torch_dtype=target_dtype,
            attn_implementation=self.config.attn_implementation,
        )
        
        # 優先從 model_path 載入對應的 processor，否則退回 processor_id
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(self.config.processor_id)
        
        print("模型載入完成")
        
    def load_test_dataset(self):
        """載入測試資料集"""
        data_path = self.config.get_data_path()
        print(f"載入測試資料集從: {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
            
        # 如果是圖文對資料，需要轉換圖片
        if self.config.data_type == "hitdata" or self.config.data_type == "mergedata":
            self.dataset = self._convert_images_in_dataset(self.dataset)
            
        print(f"載入 {len(self.dataset)} 筆測試資料")
        
    def _base64_to_pil(self, base64_str: str) -> Image.Image:
        """將 base64 字串轉為 PIL Image"""
        import base64
        from io import BytesIO
        
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))
        
    def _compress_image(self, img: Image.Image, scale_percent: int = 20, quality: int = 50) -> Image.Image:
        """壓縮圖片"""
        from io import BytesIO
        
        img = img.convert("RGB")
        width, height = img.size
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        buffer = BytesIO()
        resized_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).copy()
        
    def _convert_images_in_dataset(self, dataset):
        """將資料集中的 base64 圖片轉換為 PIL.Image"""
        print("轉換資料集中的圖片...")
        
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
                            img = self._base64_to_pil(content["image"])
                            img = self._compress_image(
                                img, 
                                scale_percent=self.config.image_scale_percent,
                                quality=self.config.image_quality
                            )
                            content["image"] = img
                        except Exception as e:
                            print(f"圖片轉換錯誤: {e}")
                            content["image"] = None
                            
        return dataset
        
    def _process_vision_info(self, messages: list[dict]) -> list[Image.Image]:
        """提取訊息中的圖片"""
        image_inputs = []
        
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            for element in content:
                if isinstance(element, dict) and (
                    "image" in element or element.get("type") == "image"
                ):
                    if "image" in element:
                        image = element["image"]
                    else:
                        image = element
                    if image is not None:
                        image_inputs.append(image.convert("RGB"))
                        
        return image_inputs
        
    def evaluate_model(self, test_count=5, save_csv=True, output_dir="experiment_result", random_select=True):
        """評估模型並保存結果
        
        Args:
            test_count: 測試樣本數量
            save_csv: 是否保存 CSV 結果
            output_dir: 輸出目錄
            random_select: 是否隨機選擇樣本
        """
        if self.model is None:
            raise ValueError("請先載入模型")
            
        if self.dataset is None:
            self.load_test_dataset()
            
        print(f"=== 模型評估：{'隨機' if random_select else '固定前'} {test_count} 筆資料 ===\n")
        
        # 選擇測試樣本
        dataset_size = len(self.dataset)
        if random_select:
            import random as _random
            test_indices = _random.sample(range(dataset_size), min(test_count, dataset_size))
        else:
            test_indices = list(range(min(test_count, dataset_size)))
            
        evaluation_results = []
        
        for i, idx in enumerate(test_indices):
            print(f"--- 測試樣本 {i+1}/{test_count} (dataset index: {idx}) ---")
            
            result = {
                'sample_index': idx,
                'test_order': i + 1,
                'data_type': '',
                'question': '',
                'model_response': '',
                'ground_truth': '',
                'evaluation_status': '',
                'error_message': '',
                'response_length': 0,
                'ground_truth_length': 0,
                'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            try:
                sample = self.dataset[idx]
                messages = sample["messages"].copy()
                
                # 檢查是否有圖片
                has_image = any(
                    isinstance(content, dict) and content.get("type") == "image"
                    for msg in messages
                    for content in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
                )
                
                data_type = '多模態 (圖片+文字)' if has_image else '純文字'
                result['data_type'] = data_type
                print(f"資料類型: {data_type}")
                
                # 提取用戶問題
                user_msg = None
                for msg in messages:
                    if msg.get("role") == "user":
                        user_msg = msg
                        break
                        
                question_text = ""
                if user_msg:
                    content = user_msg.get("content", [])
                    if isinstance(content, str):
                        question_text = content
                    elif isinstance(content, list):
                        text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                        if text_parts:
                            question_text = text_parts[0]
                            
                result['question'] = question_text
                print(f"問題: {question_text[:200]}...")
                
                # 準備推理用的 messages
                inference_messages = [msg for msg in messages if msg.get("role") != "assistant"]
                
                # 應用聊天模板
                text = self.processor.apply_chat_template(
                    inference_messages, tokenize=False, add_generation_prompt=True
                )
                
                # 處理圖片
                image_inputs = self._process_vision_info(inference_messages)
                
                # 準備輸入
                inputs = self.processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    padding=True,
                    return_tensors="pt",
                )
                
                inputs = inputs.to(self.model.device)
                
                # 生成回應
                stop_token_ids = [
                    self.processor.tokenizer.eos_token_id,
                    self.processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                ]
                
                print("模型回應:")
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        top_p=self.config.top_p,
                        do_sample=True,
                        temperature=self.config.temperature,
                        eos_token_id=stop_token_ids,
                        disable_compile=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                # 解碼輸出
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                result['model_response'] = output_text
                result['response_length'] = len(output_text)
                print(f"{output_text}\n")
                
                # 提取真實答案
                assistant_msg = None
                for msg in messages:
                    if msg.get("role") == "assistant":
                        assistant_msg = msg
                        break
                        
                if assistant_msg:
                    true_answer = assistant_msg.get("content", "")
                    if isinstance(true_answer, list):
                        true_answer = " ".join([c.get("text", "") for c in true_answer if isinstance(c, dict)])
                    result['ground_truth'] = true_answer
                    result['ground_truth_length'] = len(true_answer)
                    print(f"標準答案: {true_answer[:300]}...")
                    
                result['evaluation_status'] = 'SUCCESS'
                print("=" * 80)
                
            except Exception as e:
                result['evaluation_status'] = 'ERROR'
                result['error_message'] = str(e)
                print(f"處理樣本 {idx} 時出錯: {e}")
                print("=" * 80)
                
            evaluation_results.append(result)
            
        # 保存結果到 CSV
        if save_csv and evaluation_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            csv_file_name = f"model_evaluation_result_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_file_name)
            
            df = pd.DataFrame(evaluation_results)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n評估結果已保存至: {csv_path}")
            
        return evaluation_results
        
    def cleanup(self):
        """清理記憶體"""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()


def main(model_path):
    """主函數 - 獨立評估模式"""
    import argparse
    
    parser = argparse.ArgumentParser(description="高爾夫球 VLM 模型評估腳本")
    parser.add_argument("--model_path", type=str,default=model_path, help="訓練好的模型路徑")
    parser.add_argument("--test_count", type=int, default=5, help="測試樣本數量")
    parser.add_argument("--output_dir", type=str, default="experiment_result", help="輸出目錄")
    parser.add_argument("--random_select", action="store_true", help="隨機選擇測試樣本")
    
    args = parser.parse_args()
    
    # 創建配置和評估器
    config = TrainingConfig()
    evaluator = GolfModelEvaluator(config=config)
    
    try:
        # 載入模型和評估
        evaluator.load_model(args.model_path)
        results = evaluator.evaluate_model(
            test_count=args.test_count,
            output_dir=args.output_dir,
            random_select=args.random_select
        )
        
        print(f"\n評估完成，共測試 {len(results)} 筆資料")
        
    except Exception as e:
        print(f"評估過程中發生錯誤: {e}")
        raise
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main(model_path="/tmp/pycharm_project_979//model/google/gemma-3-4b-pttextonly-2025_07_19_1203")