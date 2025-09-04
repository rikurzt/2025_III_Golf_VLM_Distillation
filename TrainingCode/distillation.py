import base64
import io
import numpy as np
import torch
import pandas as pd
from PIL import Image
from overrides import overrides
import os

from torch.nn import MSELoss
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from trl import SFTTrainer
from transformers import is_wandb_available

if is_wandb_available():
    import wandb


def to_tensor_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def serialize_tensor_list(tensors) -> str:
    if not tensors:
        return ""
    buf = io.BytesIO()
    arrays = {f"arr_{i}": (t.squeeze(0).to(torch.float16).cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t))
              for i, t in enumerate(tensors)}
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def deserialize_tensor_list(b64str):
    if not isinstance(b64str, str) or b64str == "":
        return None
    data = base64.b64decode(b64str.encode("ascii"))
    buf = io.BytesIO(data)
    with np.load(buf, allow_pickle=False) as npz:
        keys = sorted(npz.files, key=lambda k: int(k.split("_")[1]))
        tensors = [torch.from_numpy(npz[k]).to(torch.float32) for k in keys]
    return tensors


class TeacherSignalsDataset(Dataset):
    """從 .pt 檔案目錄中讀取教師訊號的資料集。"""
    def __init__(self, signals_dir_path: str, raw_dataset):
        self.signals_dir_path = signals_dir_path
        self.raw_dataset = raw_dataset

        if not os.path.isdir(self.signals_dir_path):
            raise FileNotFoundError(f"指定的訊號目錄不存在: {self.signals_dir_path}")

        # 獲取所有以數字命名的子目錄並排序
        self.sample_dirs = sorted(
            [d for d in os.listdir(signals_dir_path) if os.path.isdir(os.path.join(signals_dir_path, d)) and d.isdigit()],
            key=lambda x: int(x)
        )
        
        num_signals = len(self.sample_dirs)
        num_raw = len(raw_dataset)

        if num_signals > num_raw:
             print(f"警告：訊號目錄數量 ({num_signals}) 大於原始資料集大小 ({num_raw})。")
        elif num_signals < num_raw:
             print(f"警告：訊號目錄數量 ({num_signals}) 小於原始資料集大小 ({num_raw})，將只使用可用的訊號。")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        # 目錄名稱即為原始資料集的索引
        sample_dir_name = self.sample_dirs[idx]
        raw_idx = int(sample_dir_name)
        
        sample_dir_path = os.path.join(self.signals_dir_path, sample_dir_name)
        
        def load_signal(filename):
            """安全地載入 .pt 檔案，若不存在則返回 None。"""
            path = os.path.join(sample_dir_path, filename)
            if os.path.exists(path):
                # 載入到 CPU 以避免在資料處理階段佔用 GPU
                return torch.load(path, map_location='cpu')
            return None

        # 獲取原始資料
        raw = self.raw_dataset[raw_idx]

        return {
            "messages": raw["messages"],
            "teacher_hidden_states": load_signal("teacher_hidden_states.pt"),
            "teacher_attentions": load_signal("teacher_attentions.pt"),
            "teacher_image_hidden_states": load_signal("teacher_image_hidden_states.pt"),
        }


def get_cor_teacher(teacher_reps, student_reps, is_attn=False):
    """
    Selects the corresponding teacher layers for the student layers based on a rounding strategy.
    This is used when the teacher model has more layers than the student model.
    """
    teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]
    teacher_layer_num = len(teacher_reps)
    student_layer_num = len(student_reps)
    
    new_teacher_reps = []

    if is_attn:
        # For attention layers (number of layers)
        if teacher_layer_num < student_layer_num:
             raise ValueError(f"Teacher attention layers ({teacher_layer_num}) < student's ({student_layer_num})")
        # We match the layers, so we use indices from 0 to student_layer_num - 1
        ratio = teacher_layer_num / student_layer_num
        for i in range(student_layer_num):
            teacher_idx = min(round(i * ratio), teacher_layer_num - 1)
            new_teacher_reps.append(teacher_reps[int(teacher_idx)])
    else:
        # For hidden states (number of layers + 1 for embeddings)
        if teacher_layer_num < student_layer_num:
             raise ValueError(f"Teacher hidden states ({teacher_layer_num}) < student's ({student_layer_num})")
        # We match the layers, so we use indices from 0 to student_layer_num - 1
        # We handle embeddings separately (always map 0 to 0)
        ratio = (teacher_layer_num - 1) / (student_layer_num - 1) if student_layer_num > 1 else 1
        for i in range(student_layer_num):
            teacher_idx = min(round(i * ratio), teacher_layer_num - 1)
            new_teacher_reps.append(teacher_reps[int(teacher_idx)])

    return new_teacher_reps


def get_kd_loss(student_reps, teacher_reps, loss_fn, is_attn=False, is_img=False):
    """
    Computes the knowledge distillation loss between student and teacher representations.
    """
    kd_loss = 0.0
    if student_reps is None or teacher_reps is None:
        return kd_loss

    if is_attn:
        for student_att, teacher_att in zip(student_reps, teacher_reps):
            '''
            if student_att.shape[1] != teacher_att.shape[1]:
                min_len = min(student_att.shape[1], teacher_att.shape[1])
                student_att = student_att[:, :min_len]
                teacher_att = teacher_att[:, :min_len]
            '''
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att), student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att), teacher_att)
            kd_loss += loss_fn(student_att, teacher_att)
            #print("att")
            #print(student_att.shape,teacher_att.shape)
    elif is_img:
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            teacher_rep = teacher_rep[0]
            '''
            if student_rep.shape[1] != teacher_rep.shape[1]:
                min_len = min(student_rep.shape[1], teacher_rep.shape[1])
                student_rep = student_rep[:, :min_len]
                teacher_rep = teacher_rep[:, :min_len]
            '''
            #print("is_img")
            #print(student_rep.shape,teacher_rep.shape)
            kd_loss += loss_fn(student_rep, teacher_rep)
    else: # for hidden states
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            '''
            if student_rep.shape[1] != teacher_rep.shape[1]:
                min_len = min(student_rep.shape[1], teacher_rep.shape[1])
                student_rep = student_rep[:, :min_len]
                teacher_rep = teacher_rep[:, :min_len]
            '''
            #print("hidden states")
            #print(student_rep.shape,teacher_rep.shape)
            kd_loss += loss_fn(student_rep, teacher_rep)

    return kd_loss

class DistillSTFTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.distill_weight = kwargs.pop('distill_weight', 1.0)
        super().__init__(*args, **kwargs)

    def compute_hidden_states_loss(self, student_hidden_states, teacher_hidden_states,loss_fn):
        if teacher_hidden_states is None:
            return 0.0
        # Align teacher and student hidden states
        teacher_hidden_states = get_cor_teacher(teacher_hidden_states, student_hidden_states, is_attn=False)
        return get_kd_loss(student_hidden_states, teacher_hidden_states, loss_fn, is_attn=False, is_img=False)

    def compute_attentions_loss(self, student_attentions, teacher_attentions,loss_fn):
        if teacher_attentions is None:
            return 0.0
        # Align teacher and student attentions
        teacher_attentions = get_cor_teacher(teacher_attentions, student_attentions, is_attn=True)
        return get_kd_loss(student_attentions, teacher_attentions, loss_fn, is_attn=True, is_img=False)

    def compute_image_hidden_states_loss(self, student_image_hidden_states, teacher_image_hidden_states,loss_fn):
        if teacher_image_hidden_states is None or student_image_hidden_states is None:
            return 0.0
        # Vision towers are identical, no layer alignment needed
        return get_kd_loss(student_image_hidden_states, teacher_image_hidden_states, loss_fn, is_attn=False, is_img=True)

    @overrides()
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        # Pop the teacher's outputs from the inputs
        teacher_hidden_states = inputs.pop("teacher_hidden_states", None)
        teacher_attentions = inputs.pop("teacher_attentions", None)
        teacher_image_hidden_states = inputs.pop("teacher_image_hidden_states", None)

        # Compute the original loss from SFTTrainer
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True,num_items_in_batch=num_items_in_batch)

        # Get student's internal states
        student_hidden_states = outputs.hidden_states
        student_attentions = outputs.attentions
        student_image_hidden_states = outputs.image_hidden_states

        
        # Compute the distillation loss
        mse_loss = MSELoss()
        hidden_states_loss = self.compute_hidden_states_loss(student_hidden_states, teacher_hidden_states, mse_loss)
        attentions_loss = self.compute_attentions_loss(student_attentions, teacher_attentions, mse_loss)
        image_hidden_states_loss = self.compute_image_hidden_states_loss(student_image_hidden_states, teacher_image_hidden_states, mse_loss)
        
        # Combine the losses (example: simple addition, could be weighted)
        distill_loss = hidden_states_loss + attentions_loss + image_hidden_states_loss

        # You can weigh the original loss and the distillation loss
        # Example: loss = 0.4 * loss + 0.6 * distill_loss
        loss += self.distill_weight * distill_loss

        return (loss, outputs) if return_outputs else loss


def create_distill_data_collator(processor):

    def process_vision_info(messages: list[dict]) -> list[Image.Image]:
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

    def collate_fn(examples):
        texts = []
        images = []
        teacher_hidden_states_list = []
        teacher_attentions_list = []
        teacher_image_hidden_states_list = []

        # 1. Extract data from each sample
        for example in examples:
            image_inputs = process_vision_info(example["messages"])
            text = processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            # 保持每樣本的影像列表，不要攤平成整個 batch 一個列表
            if len(image_inputs) == 0:
                images.append(None)
            else:
                images.append(image_inputs)

            # Extract teacher outputs, ensuring they are not None
            if "teacher_hidden_states" in example and example["teacher_hidden_states"] is not None:
                teacher_hidden_states_list.append(example["teacher_hidden_states"])
            if "teacher_attentions" in example and example["teacher_attentions"] is not None:
                teacher_attentions_list.append(example["teacher_attentions"])
            if "teacher_image_hidden_states" in example and example["teacher_image_hidden_states"] is not None:
                teacher_image_hidden_states_list.append(example["teacher_image_hidden_states"])

        # 2. Process and tokenize text and images
        # 若整個 batch 都沒有圖片，傳 None；否則按樣本傳入各自的圖片列表
        if any(img_list for img_list in images):
            images_for_processor = [img_list if img_list else None for img_list in images]
            batch = processor(
                text=texts,
                images=images_for_processor,
                return_tensors="pt",
                padding=True
            )
        else:
            batch = processor(text=texts, images=None, return_tensors="pt", padding=True)

        # 3. Create labels, masking where necessary
        labels = batch["input_ids"].clone()
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100
        batch["labels"] = labels

        # 4. Pad and stack the teacher's outputs
        if teacher_hidden_states_list:
            padded_hidden_states = []
            # Transpose and pad each layer (每層為 [S, D]，pad 成 [B, S_max, D])
            for layer_tensors in zip(*teacher_hidden_states_list):
                padded_layer = pad_sequence(list(layer_tensors), batch_first=True, padding_value=0.0)
                padded_hidden_states.append(padded_layer)
            batch["teacher_hidden_states"] = tuple(padded_hidden_states)

        if teacher_attentions_list:
            padded_attentions = []
            # 注意：每層 attention 反序列化後為 [H, S, S]，pad_sequence 會在第一維 (H) pad。
            # 這裡先保持簡單策略，後續在 loss 中會以最小序列長度裁切對齊。
            for layer_tensors in zip(*teacher_attentions_list):
                padded_layer = pad_sequence(list(layer_tensors), batch_first=True, padding_value=0.0)
                padded_attentions.append(padded_layer)
            batch["teacher_attentions"] = tuple(padded_attentions)

        if teacher_image_hidden_states_list:
            padded_image_hidden_states = []
            # 每層影像隱狀態為 [S_img, D]，pad 成 [B, S_img_max, D]
            for layer_tensors in zip(*teacher_image_hidden_states_list):
                padded_layer = pad_sequence(list(layer_tensors), batch_first=True, padding_value=0.0)
                padded_image_hidden_states.append(padded_layer)
            batch["teacher_image_hidden_states"] = tuple(padded_image_hidden_states)

        return batch
    return collate_fn 