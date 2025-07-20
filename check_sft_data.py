import json

# 讀取 SFT 訓練資料
with open('dataset/0513_SFTDataset/hitdata/sft_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 檢查資料結構
print(f"資料總數: {len(data)}")
print("\n第一筆資料結構:")
print(json.dumps(data[0], indent=2, ensure_ascii=False))

print("\n檢查是否有 image 欄位:")
if 'image' in data[0]:
    print("有 image 欄位")
else:
    print("沒有 image 欄位")
    
# 檢查 messages 結構
if 'messages' in data[0]:
    print("\nmessages 結構:")
    for i, msg in enumerate(data[0]['messages']):
        print(f"Message {i}: {msg['role']}")
        if 'content' in msg:
            print(f"  內容類型: {type(msg['content'])}")
            if isinstance(msg['content'], list):
                for j, content in enumerate(msg['content']):
                    print(f"    {j}: {content.get('type', 'unknown')}") 