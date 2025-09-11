import base64
import io
import numpy as np
import torch
import pandas as pd
from PIL import Image
from overrides import overrides
import os
from torchvision import transforms

import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from trl import SFTTrainer
from transformers import is_wandb_available

if is_wandb_available():
    import wandb


class StudentModelWithProjector(torch.nn.Module):
    def __init__(self, model, projector):
        super().__init__()
        self.model = model
        self.projector = projector

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.model, name):
                return getattr(self.model, name)
            raise


def to_tensor_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


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

            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att), student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att), teacher_att)
            if teacher_att.dim() == student_att.dim() + 1 and teacher_att.shape[1] == 1:
                teacher_att = teacher_att.squeeze(1)
            kd_loss += loss_fn(student_att, teacher_att.to(student_att.dtype))
            #print("att")
            #print(student_att.shape,teacher_att.shape)
    elif is_img:
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            # teacher_rep = teacher_rep[0] # This is likely incorrect for batch_size > 1

            if teacher_rep.dim() == student_rep.dim() + 1 and teacher_rep.shape[1] == 1:
                teacher_rep = teacher_rep.squeeze(1)
            #print("is_img")
            #print(student_rep.shape,teacher_rep.shape)
            kd_loss += loss_fn(student_rep, teacher_rep.to(student_rep.dtype))
    else: # for hidden states
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):

            if teacher_rep.dim() == student_rep.dim() + 1 and teacher_rep.shape[1] == 1:
                teacher_rep = teacher_rep.squeeze(1)

            #print("hidden states")
            #print(student_rep.shape,teacher_rep.shape)
            kd_loss += loss_fn(student_rep, teacher_rep.to(student_rep.dtype))

    return kd_loss

class DistillSTFTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None,
                 do_hidden_states_loss=True, do_attentions_loss=True, do_image_hidden_states_loss=True,
                 distill_rate=1.0, hidden_states_last_only=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_rate = distill_rate
        self.teacher_model = teacher_model

        if self.teacher_model is not None:
            self.teacher_model.eval()

        self.do_hidden_states_loss = do_hidden_states_loss
        self.do_attentions_loss = do_attentions_loss
        self.do_image_hidden_states_loss = do_image_hidden_states_loss
        self.hidden_states_last_only = hidden_states_last_only

    def compute_hidden_states_loss(self, student_hidden_states, teacher_hidden_states):
        """
        助教模型的隱藏狀態經過投影後，與學生模型的隱藏狀態進行對齊和LOSS計算，採用 MSE Loss全層平均。
        """
        if teacher_hidden_states is None or student_hidden_states is None:
            return 0.0

        # 按比例選教師層對齊學生層
        teacher_hidden_states = get_cor_teacher(teacher_hidden_states, student_hidden_states, is_attn=False)

        # 確保 dtype 對齊到學生參數 dtype（只抓一次參考參數即可）
        student_param = next(self.model.parameters())
        target_dtype = student_param.dtype

        # 僅計算最後一層
        if self.hidden_states_last_only:
            s_h = student_hidden_states[-1].to(dtype=target_dtype)
            t_h = teacher_hidden_states[-1].to(dtype=target_dtype)
            t_h_logits = t_h @ self.teacher_model.lm_head.weight.T  # 用教師 head 得到 logits
            # --- Teacher → Student space ---
            t2s_h = self.model.projector["t2s"](t_h)  # 投影教師 hidden 到學生空間
            t2s_h = t2s_h.to(dtype=target_dtype)
            # distillation loss (學生 logits vs 投影教師 logits)
            with torch.no_grad():
                t2s_logits = t2s_h @ self.model.lm_head.weight.detach().T  # 用學生 head 得到 logits
                student_logits = (s_h @ self.model.lm_head.weight.T).to(dtype=target_dtype)
            t2s_kd = F.mse_loss(student_logits, t2s_logits.detach().to(dtype=target_dtype))
            th_kd = F.mse_loss(t2s_logits, t_h_logits.detach().to(dtype=target_dtype))
            return (t2s_kd + th_kd) / 2

        total_loss = 0.0
        num_layers = 0
        for s_h, t_h in zip(student_hidden_states, teacher_hidden_states):
            # 確保 dtype 對齊到學生參數 dtype
            s_h = s_h.to(dtype=target_dtype)
            t_h = t_h.to(dtype=target_dtype)
            t_h_logits = t_h @ self.teacher_model.lm_head.weight.T  # 用教師 head 得到 logits
            # --- Teacher → Student space ---
            t2s_h = self.model.projector["t2s"](t_h)  # 投影教師 hidden 到學生空間
            t2s_h = t2s_h.to(dtype=target_dtype)
            # distillation loss (學生 logits vs 投影教師 logits)
            with torch.no_grad():
                t2s_logits = t2s_h @ self.model.lm_head.weight.detach().T  # 用學生 head 得到 logits
                student_logits = (s_h @ self.model.lm_head.weight.T).to(dtype=target_dtype)
            t2s_kd = F.mse_loss(student_logits, t2s_logits.detach().to(dtype=target_dtype))
            th_kd = F.mse_loss(t2s_logits, t_h_logits.detach().to(dtype=target_dtype))
            total_loss += (t2s_kd + th_kd) / 2
            num_layers += 1

        if num_layers > 0:
            total_loss = total_loss / num_layers

        return total_loss
    def compute_attentions_loss(self, student_attentions, teacher_attentions,loss_fn):
        if teacher_attentions is None:
            return 0.0
        # Align teacher and student attentions
        teacher_attentions = get_cor_teacher(teacher_attentions, student_attentions, is_attn=True)
        return get_kd_loss(student_attentions, teacher_attentions, loss_fn, is_attn=True, is_img=False)

    def compute_image_hidden_states_loss(self, student_image_hidden_states, teacher_image_hidden_states,loss_fn):
        if teacher_image_hidden_states is None or student_image_hidden_states is None:
            return 0.0
        # 動態對齊 projector 維度，避免 hidden 維度不一致
        student_param = next(self.model.parameters())
        target_dtype = student_param.dtype

        s_h = student_image_hidden_states.to(dtype=target_dtype)
        t_h = student_image_hidden_states.to(dtype=target_dtype)

        img_ls = F.mse_loss(s_h, t_h)
        return img_ls

    from torch.nn import MSELoss
    from overrides import overrides
    import torch

    @overrides()
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 準備教師輸入
        teacher_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
        }
        teacher_inputs = {k: v for k, v in teacher_inputs.items() if v is not None}

        # 決定是否需要 hidden/attn
        output_hs = self.do_hidden_states_loss
        output_attn = self.do_attentions_loss
        output_img_hs = self.do_image_hidden_states_loss
        #image hidden states 要自己hook
        with torch.no_grad():
            self.teacher_model.to(model.device)
            if hasattr(model.config, "vision_config"):
                model.config.vision_config.output_hidden_states = False
            teacher_outputs = self.teacher_model.generate(
                **teacher_inputs,
                return_dict_in_generate=True,
                output_hidden_states=output_hs,
                output_attentions=output_attn,
                do_sample = False,
                max_new_tokens = 1024,
            )
            teacher_outputs_2 = self.teacher_model(#需優化
                **teacher_inputs,
                output_hidden_states=output_hs,
                output_attentions=output_attn,
                do_sample=False,
                max_new_tokens=1024,
            )



        # 取出最後生成的文字
        teacher_text = self.processing_class.batch_decode(
            teacher_outputs.sequences
        )
        teacher_text2 = self.processing_class.batch_decode(
            inputs.get("input_ids")
        )

        #for i in range(len(teacher_outputs.hidden_states)):
        #    print(teacher_outputs.hidden_states[i][0].shape)#why只有地一個是[1,1,3840]?


        teacher_hidden_states = teacher_outputs.hidden_states[0] if output_hs else None
        teacher_attentions = teacher_outputs.attentions[0] if output_attn else None
        teacher_image_hidden_states = teacher_outputs_2.image_hidden_states if output_img_hs else None

        teacher_retonkenize = self.processing_class.tokenizer(
            teacher_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=inputs["input_ids"].shape[1],
        ).to(model.device)


        student_inputs = {
            "input_ids": teacher_retonkenize['input_ids'],
            "attention_mask": teacher_retonkenize['attention_mask'],
            "pixel_values": inputs.get("pixel_values"),
            "labels": teacher_retonkenize['input_ids'],
        }

        loss, outputs = super().compute_loss(model, student_inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        student_hidden_states = outputs.hidden_states
        student_attentions = outputs.attentions
        student_image_hidden_states = outputs.image_hidden_states


        # 計算 distillation loss
        mse_loss = MSELoss()
        hidden_states_loss = 0.0
        attentions_loss = 0.0
        image_hidden_states_loss = 0.0

        if self.do_hidden_states_loss:
            hidden_states_loss = self.compute_hidden_states_loss(student_hidden_states,
                                                                 teacher_hidden_states)

        if self.do_attentions_loss:
            attentions_loss = self.compute_attentions_loss(student_attentions, None, mse_loss)  # attn 要不要算 generate?

        if self.do_image_hidden_states_loss:
            image_hidden_states_loss = self.compute_image_hidden_states_loss(student_image_hidden_states, None,
                                                                             mse_loss)

        distill_loss = hidden_states_loss + attentions_loss + image_hidden_states_loss
        loss = self.distill_rate * distill_loss + (1 - self.distill_rate) * loss

        # 清理顯存
        if self.teacher_model is not None and 'teacher_outputs' in locals():
            del teacher_outputs
            del teacher_text
            torch.cuda.empty_cache()

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

        return batch
    return collate_fn 