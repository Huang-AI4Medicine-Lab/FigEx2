import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from transformers.modeling_outputs import CausalLMOutputWithPast


PROMPT_COMPOUND = (
    "You are given a scientific compound figure.\n"
    "Task: detect subfigures (panels) and, for each detected subfigure that shows a visible alphabetic label (A–Z or a–z), "
    "write exactly one short, scientific caption.\n"
    "Formatting rules:\n"
    "1) Output one line per subfigure in ascending label order (A, B, C, ...).\n"
    '2) Use uppercase labels and the exact format: "A: <caption>".\n'
    "3) After listing all subfigure captions, output a single [DET] token on a NEW line.\n"
    "Return ONLY the caption lines followed by the final [DET]; no extra text."
)


def _letter_from_idx(idx: int) -> Optional[str]:

    if 0 <= idx <= 25:
        return chr(ord("A") + idx)
    return None


def _parse_subcaptions_lines(raw_text: str) -> Dict[str, str]:


    mapping: Dict[str, str] = {}
    lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
    for ln in lines:
        if ln == "[DET]":
            break
        m = re.match(r"^\s*([A-Za-z])\s*:\s*(.+?)\s*$", ln)
        if m:
            letter = m.group(1).upper()
            txt = m.group(2).strip()
            if txt:
                mapping[letter] = txt
    return mapping


class FigEx2ForCausalLM(Qwen3VLForConditionalGeneration):


    def __init__(self, config, **kwargs):

        self.det_token_idx = kwargs.pop("det_token_idx", -1)
        super().__init__(config)

        try:
            self.config.use_cache = False
        except Exception:
            pass

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        mm_token_type_ids=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if mm_token_type_ids is not None:
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
        return model_inputs


    @torch.no_grad()
    def evaluate(
        self,
        qwen_processor: AutoProcessor,
        tokenizer,
        image_path: str,
        device: torch.device,
        max_new_tokens: int = 512,
        prompt: str = PROMPT_COMPOUND,
        return_generation_details: bool = False,
    ):


        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            chat_prompt = qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_prompt = prompt


        pil_img = Image.open(image_path).convert("RGB")


        gen_inputs = qwen_processor(
            text=[chat_prompt],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        )


        for k, v in list(gen_inputs.items()):
            if torch.is_tensor(v):
                gen_inputs[k] = v.to(device)


        prompt_mm_token_type_ids = gen_inputs.get("mm_token_type_ids")


        if "attention_mask" not in gen_inputs:
            raise RuntimeError("Processor output missing attention_mask; cannot build the full-sequence forward.")
        prompt_input_width = int(gen_inputs["input_ids"].shape[1])


        det_token_id = tokenizer.convert_tokens_to_ids("[DET]")
        if det_token_id is None or det_token_id < 0 or det_token_id == tokenizer.unk_token_id:
            raise RuntimeError("Tokenizer does not contain [DET] token (or it maps to unk).")


        eos_ids = [int(tokenizer.eos_token_id)]
        if int(det_token_id) not in eos_ids:
            eos_ids.append(int(det_token_id))


        gen_outputs = self.generate(
            **gen_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )

        sequences = gen_outputs.sequences
        seq = sequences[0]


        gen_ids = seq[prompt_input_width:]
        raw_full = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

        lines = [ln.rstrip() for ln in raw_full.splitlines() if ln.strip()]
        raw_text = "\n".join(lines)


        det_pos_rel = (gen_ids == int(det_token_id)).nonzero(as_tuple=True)[0]
        if det_pos_rel.numel() > 0:
            t_det_rel = int(det_pos_rel[0].item())
            need_append_det = False
        else:
            t_det_rel = gen_ids.numel()
            need_append_det = True

        if need_append_det and t_det_rel > 0:
            trailing_special_ids = {
                int(token_id)
                for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
                if token_id is not None
            }
            while t_det_rel > 0 and int(gen_ids[t_det_rel - 1].item()) in trailing_special_ids:
                t_det_rel -= 1

        if t_det_rel < 0 or t_det_rel > gen_ids.numel():
            raise RuntimeError(f"[DET] position {t_det_rel} out of range for generated length {gen_ids.numel()}.")

        text_gen_ids = gen_ids[:t_det_rel].detach().cpu().tolist()
        text_before_det = tokenizer.decode(text_gen_ids, skip_special_tokens=False)

        token_char_offsets = []
        cursor = 0
        for token_id in text_gen_ids:
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            start = cursor
            cursor += len(piece)
            token_char_offsets.append((start, cursor))


        caption_gen_ids = gen_ids[:t_det_rel]
        if need_append_det:
            det_gen_id = torch.tensor([int(det_token_id)], device=seq.device, dtype=seq.dtype)
            if not any(line.strip() == "[DET]" for line in lines):
                raw_text = "\n".join([*lines, "[DET]"])
        else:
            det_gen_id = gen_ids[t_det_rel : t_det_rel + 1]

        feature_gen_ids = torch.cat([caption_gen_ids, det_gen_id], dim=0).unsqueeze(0)
        full_input_ids = torch.cat(
            [sequences[:, :prompt_input_width], feature_gen_ids],
            dim=1,
        )
        full_attention_mask = torch.cat(
            [
                gen_inputs["attention_mask"],
                torch.ones(
                    (1, feature_gen_ids.shape[1]),
                    device=gen_inputs["attention_mask"].device,
                    dtype=gen_inputs["attention_mask"].dtype,
                ),
            ],
            dim=1,
        )
        full_mm_token_type_ids = None
        if prompt_mm_token_type_ids is not None:


            full_mm_token_type_ids = torch.cat(
                [
                    prompt_mm_token_type_ids,
                    torch.zeros(
                        (1, feature_gen_ids.shape[1]),
                        device=prompt_mm_token_type_ids.device,
                        dtype=prompt_mm_token_type_ids.dtype,
                    ),
                ],
                dim=1,
            )

        full_model_kwargs = dict(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        if "pixel_values" in gen_inputs:
            full_model_kwargs["pixel_values"] = gen_inputs["pixel_values"]
        if "image_grid_thw" in gen_inputs:
            full_model_kwargs["image_grid_thw"] = gen_inputs["image_grid_thw"]
        if "pixel_values_videos" in gen_inputs:
            full_model_kwargs["pixel_values_videos"] = gen_inputs["pixel_values_videos"]
        if "video_grid_thw" in gen_inputs:
            full_model_kwargs["video_grid_thw"] = gen_inputs["video_grid_thw"]
        if full_mm_token_type_ids is not None:
            full_model_kwargs["mm_token_type_ids"] = full_mm_token_type_ids

        full_outputs = self.model(**full_model_kwargs)
        full_last_hidden = full_outputs.last_hidden_state


        t_det = prompt_input_width + t_det_rel
        if t_det >= full_last_hidden.shape[1]:

            raise RuntimeError(
                f"[DET] absolute position {t_det} out of range for full hidden length {full_last_hidden.shape[1]}."
            )

        det_vec = full_last_hidden[0, t_det, :]
        det_feats = det_vec.unsqueeze(0).unsqueeze(1)


        ext_text_feats = full_last_hidden[:, prompt_input_width:t_det, :].contiguous()

        if return_generation_details:
            return raw_text, det_feats, ext_text_feats, {
                "text_before_det": text_before_det,
                "token_char_offsets": token_char_offsets,
                "text_token_ids": text_gen_ids,
            }

        return raw_text, det_feats, ext_text_feats


    def model_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
        tokenizer,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        entity_token_indices: Optional[torch.LongTensor] = None,
    ):


        if input_ids is None or attention_mask is None or labels is None:
            raise ValueError("model_forward requires input_ids, attention_mask, and labels.")
        if pixel_values is None:
            raise ValueError("pixel_values is required in your current design (image is always used).")


        model_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw
        if mm_token_type_ids is not None:
            model_kwargs["mm_token_type_ids"] = mm_token_type_ids

        outputs = self.model(**model_kwargs)
        last_hidden = outputs.last_hidden_state


        logits = self.lm_head(last_hidden)


        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        lm_out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=None,
            attentions=None,
        )


        B, L, D = last_hidden.shape
        det_id = int(self.det_token_idx)

        det_feats_list: List[torch.Tensor] = []
        ext_feats_list: List[torch.Tensor] = []
        ext_indices_list: List[torch.Tensor] = []

        for b in range(B):
            lab = labels[b]

            non_ignore = (lab != -100).nonzero(as_tuple=True)[0]
            if non_ignore.numel() == 0:

                print("[WARN] Sample has no non -100 labels; skipping batch.")
                return None, None, None, None
            gen_start = int(non_ignore.min().item())

            det_pos_all = (lab == det_id).nonzero(as_tuple=True)[0]
            det_pos = int(det_pos_all[-1].item()) if det_pos_all.numel() > 0 else (lab.size(0) - 1)

            det_vec = last_hidden[b, det_pos, :]
            det_feats_list.append(det_vec)

            ext = None
            ext_idx = None
            if entity_token_indices is not None:
                idx = entity_token_indices[b]
                idx = idx[(idx >= 0) & (idx < L)]
                if idx.numel() > 0:
                    ext = last_hidden[b, idx, :].contiguous()
                    ext_idx = idx.contiguous()
            if ext is None:
                ext = last_hidden[b, gen_start:det_pos, :].contiguous()
                ext_idx = torch.arange(gen_start, det_pos, device=last_hidden.device, dtype=torch.long)
            ext_feats_list.append(ext)
            ext_indices_list.append(ext_idx)

        det_feats = torch.stack(det_feats_list, dim=0).unsqueeze(1)

        max_t = max(x.size(0) for x in ext_feats_list) if ext_feats_list else 0
        if max_t == 0:
            ext_text_feats = None
            ext_token_indices = None
        else:
            ext_text_feats = last_hidden.new_zeros((B, max_t, D))
            ext_token_indices = torch.full((B, max_t), -1, dtype=torch.long, device=last_hidden.device)
            for b in range(B):
                t = ext_feats_list[b].size(0)
                if t > 0:
                    ext_text_feats[b, :t, :] = ext_feats_list[b]
                    ext_token_indices[b, :t] = ext_indices_list[b]

        return lm_out, det_feats, ext_text_feats, ext_token_indices
