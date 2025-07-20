"""
多模態SFT訓練解決方案 - 同時支援有圖片和無圖片的數據

此解決方案展示如何修改現有的SFT流程，讓它能夠同時處理：
1. 有圖片的多模態數據 (如: sft_training_data.json)
2. 純文字的問答對數據 (如: qa_pairs_sft.json)
"""

import json
from PIL import Image
from typing import List, Dict, Any, Optional

def load_datasets(image_data_path: str, text_data_path: str):
    """載入圖片和文字數據集"""
    
    # 載入圖片數據集
    with open(image_data_path, "r", encoding="utf-8") as f:
        image_dataset = json.load(f)
    
    # 載入純文字數據集
    with open(text_data_path, "r", encoding="utf-8") as f:
        text_dataset = json.load(f)
    
    print(f"圖片數據集: {len(image_dataset)} 筆")
    print(f"文字數據集: {len(text_dataset)} 筆")
    
    # 合併數據集
    combined_dataset = image_dataset + text_dataset
    print(f"合併後總數據量: {len(combined_dataset)} 筆")
    
    return combined_dataset

def process_vision_info(messages: List[Dict]) -> List[Image.Image]:
    """處理訊息中的圖片信息"""
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

def improved_collate_fn(examples, processor):
    """
    改進的數據整理函數，支援混合模態數據
    
    關鍵改進：
    1. 檢測每個樣本是否包含圖片
    2. 分別處理有圖片和無圖片的樣本
    3. 只有在有圖片時才處理圖片相關的token
    """
    texts = []
    images = []
    
    for example in examples:
        # 提取圖片
        image_inputs = process_vision_info(example["messages"])
        
        # 生成對話文本
        text = processor.apply_chat_template(
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
        # 處理混合模態數據
        processed_images = []
        for img_list in images:
            if img_list is None:
                continue  # 跳過沒有圖片的樣本
            else:
                processed_images.extend(img_list)
        
        # 使用processor處理文字和圖片
        batch = processor(
            text=texts, 
            images=processed_images if processed_images else None, 
            return_tensors="pt", 
            padding=True
        )
    else:
        # 純文字處理
        batch = processor(text=texts, images=None, return_tensors="pt", padding=True)

    # 創建標籤
    labels = batch["input_ids"].clone()

    # 只有當有圖片時才處理圖片token
    if has_images and any(img is not None for img in images):
        try:
            # 查找圖片token ID
            if "boi_token" in processor.tokenizer.special_tokens_map:
                image_token_id = processor.tokenizer.convert_tokens_to_ids(
                    processor.tokenizer.special_tokens_map["boi_token"]
                )
                labels[labels == image_token_id] = -100
            
            # 其他圖片相關的token
            labels[labels == 262144] = -100
            
        except (KeyError, AttributeError):
            # 如果沒有圖片相關的特殊token，跳過
            pass
    
    # 遮蔽填充token
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    return batch

def setup_multimodal_training():
    """設置多模態訓練的完整示例"""
    
    # 1. 載入數據
    image_data_path = "./dataset/0513_SFTDataset/hitdata/sft_training_data.json"
    text_data_path = "./dataset/0513_SFTDataset/text/qa_pairs_sft.json"
    combined_dataset = load_datasets(image_data_path, text_data_path)
    
    # 2. 設置訓練配置
    training_config = {
        "output_dir": "./gemma-3-4b-golf-multimodal-SFT",
        "bf16": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "max_seq_length": 2048,
        "logging_steps": 25,
        "save_steps": 50,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }
    
    print("多模態訓練配置完成！")
    print("主要特點：")
    print("1. ✅ 支援圖片+文字的多模態數據")
    print("2. ✅ 支援純文字的問答對數據") 
    print("3. ✅ 自動檢測並處理不同類型的數據")
    print("4. ✅ 優化的token遮蔽策略")
    
    return combined_dataset, training_config

# 使用範例
if __name__ == "__main__":
    # 設置多模態訓練
    dataset, config = setup_multimodal_training()
    
    # 在實際使用時，你可以這樣創建訓練器：
    """
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**config),
        train_dataset=dataset,  # 使用合併後的數據集
        data_collator=lambda x: improved_collate_fn(x, processor),
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )
    """ 