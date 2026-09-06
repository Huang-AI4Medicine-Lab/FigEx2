#!/usr/bin/env python


import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from peft import PeftModel
import peft.peft_model as peft_model_module
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

from model.FigEx2_dab_detr import FigEx2ForCausalLM, PROMPT_COMPOUND
from model.dab_detr.modeling_dab_detr_figex2 import DabDetrForObjectDetection


BOX_MAX_N = 300
_SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>|\[/?[A-Z]+\]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BioSci-Fig-Cap FigEx2 inference (no metric evaluation)")
    p.add_argument("--test-json", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--peft-dir", required=True)
    p.add_argument("--dabdetr-dir", required=True)
    p.add_argument("--lm-base-id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--det-base-id", default="IDEA-Research/dab-detr-resnet-50")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--score-threshold", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=BOX_MAX_N)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--save-unfiltered-detections", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--few-shot", type=int, default=0, choices=[0, 1, 2])
    return p.parse_args()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_test_list(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported test JSON format: {json_path}")


def resolve_image_path(image_root: Path, file_name: str) -> Optional[Path]:
    candidates = [
        image_root / file_name,
        image_root / f"test__{file_name}",
        image_root / Path(file_name).name,
        image_root / f"test__{Path(file_name).name}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def extract_assistant_segment(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    idx = lower.find("assistant")
    if idx != -1:
        text = text[idx + len("assistant") :]
    return text.lstrip(" :\n\r\t")


def clean_caption_text(text: str) -> str:
    text = _SPECIAL_TOKEN_RE.sub(" ", text or "")
    text = text.replace("�", " ")
    text = re.sub(r"^[\-\*\u2022]+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r\"'、，。；;:：")


def parse_subcaptions(raw_text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    text = extract_assistant_segment(raw_text)
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
        if line == "[DET]":
            break
        match = re.match(r"^\s*([A-Za-z])\s*:\s*(.+?)\s*$", line)
        if not match:
            continue
        caption = clean_caption_text(match.group(2))
        if caption:
            mapping[match.group(1).upper()] = caption
    return mapping


FEWSHOT_LABELS = ["A", "B"]


def label_to_class_id(label: str) -> int:
    label = (label or "").strip().upper()
    if len(label) != 1 or not ("A" <= label <= "Z"):
        return 0
    return ord(label) - ord("A")


def bbox_xywh_to_xyxy(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x, y, w, h = [float(v) for v in bbox[:4]]
    return [x, y, x + w, y + h]


def coerce_xyxy(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    return [x0, y0, x1, y1]


def extract_gt_subcaptions_and_bboxes(item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    gt: Dict[str, Dict[str, Any]] = {}
    candidates = []
    for key in [
        "split",
        "splits",
        "subcaptions",
        "sub_captions",
        "sub_caps",
        "subfigures",
        "sub_figures",
        "subfigs",
        "panels",
        "panel_annotations",
        "annotations",
        "gt_subcaptions",
        "gt",
        "labels",
    ]:
        if key in item:
            candidates.append(item[key])

    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        for label, value in obj.items():
            if not isinstance(label, str) or len(label.strip()) != 1 or not label.strip().isalpha():
                continue
            label = label.strip().upper()
            if isinstance(value, str):
                gt[label] = {"caption": value, "bbox_xyxy": None}
            elif isinstance(value, dict):
                caption = value.get("caption") or value.get("text") or value.get("subcaption")
                bbox = value.get("bbox_xyxy") or value.get("bbox")
                bbox_xyxy = coerce_xyxy(bbox) or bbox_xywh_to_xyxy(bbox)
                if caption is not None:
                    gt[label] = {"caption": str(caption), "bbox_xyxy": bbox_xyxy}
        if gt:
            return gt

    for obj in candidates:
        if not isinstance(obj, list):
            continue
        for entry in obj:
            if not isinstance(entry, dict):
                continue
            label = (
                entry.get("sub-image_name")
                or entry.get("sub_image_name")
                or entry.get("label")
                or entry.get("panel")
                or entry.get("id")
                or entry.get("name")
            )
            if not isinstance(label, str) or not label.strip():
                continue
            label = label.strip()[0].upper()
            if not ("A" <= label <= "Z"):
                continue
            caption = entry.get("caption") or entry.get("text") or entry.get("subcaption")
            bbox = entry.get("bbox_xyxy") or entry.get("bbox")
            bbox_xyxy = coerce_xyxy(bbox) or bbox_xywh_to_xyxy(bbox)
            if bbox_xyxy is None and all(k in entry for k in ["x1", "y1", "x2", "y2"]):
                bbox_xyxy = [float(entry["x1"]), float(entry["y1"]), float(entry["x2"]), float(entry["y2"])]
            if caption is not None:
                gt[label] = {"caption": str(caption), "bbox_xyxy": bbox_xyxy}
        if gt:
            return gt
    return gt


def build_fewshot_prompt(fewshot_caps: Dict[str, str]) -> str:
    if not fewshot_caps:
        return PROMPT_COMPOUND
    prompt = PROMPT_COMPOUND + "\n"
    for label in FEWSHOT_LABELS:
        caption = fewshot_caps.get(label)
        if caption:
            prompt += f"{label}: {caption}\n"
    return prompt


def inject_provided_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: Optional[torch.Tensor],
    provided_boxes: Dict[int, List[float]],
    max_det: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if not provided_boxes:
        return boxes, scores, labels
    pred_by_class: Dict[int, Dict[str, Any]] = {}
    if labels is not None and len(labels) == len(scores):
        for i in range(min(len(scores), max_det)):
            cls_id = int(labels[i].item())
            pred_by_class[cls_id] = {
                "score": float(scores[i].item()),
                "bbox_xyxy": [float(v) for v in boxes[i].tolist()],
            }
    for cls_id, bbox in provided_boxes.items():
        pred_by_class[cls_id] = {"score": 1.0, "bbox_xyxy": [float(v) for v in bbox]}
    items = [(cls_id, value["score"], value["bbox_xyxy"]) for cls_id, value in pred_by_class.items()]
    items.sort(key=lambda x: (-float(x[1]), int(x[0])))
    items = items[:max_det]
    if not items:
        return boxes[:0], scores[:0], torch.empty((0,), dtype=torch.long)
    return (
        torch.tensor([item[2] for item in items], dtype=torch.float32),
        torch.tensor([item[1] for item in items], dtype=torch.float32),
        torch.tensor([item[0] for item in items], dtype=torch.long),
    )


def build_lm(lm_base_id: str, peft_dir: Path, device: torch.device, dtype: torch.dtype):
    qwen_processor = AutoProcessor.from_pretrained(lm_base_id, trust_remote_code=True, use_fast=True)


    tokenizer = AutoTokenizer.from_pretrained(str(peft_dir), use_fast=True, trust_remote_code=True)
    qwen_processor.tokenizer = tokenizer

    det_token_id = tokenizer.convert_tokens_to_ids("[DET]")
    if det_token_id is None or det_token_id < 0 or det_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"[DET] token not found in tokenizer from {peft_dir}")

    base_model = FigEx2ForCausalLM.from_pretrained(
        lm_base_id,
        dtype=dtype,
        det_token_idx=int(det_token_id),
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    if base_model.get_input_embeddings().num_embeddings != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))

    backbone_adapter = peft_dir / "backbone_adapter"
    original_infer_device = peft_model_module.infer_device
    peft_model_module.infer_device = lambda: "cpu"
    try:
        if not (backbone_adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Missing required FigEx2 component at {backbone_adapter}. "
                "Download the complete FigEx2 weight bundle."
            )
        backbone_policy = PeftModel.from_pretrained(base_model, str(backbone_adapter), is_trainable=False)
        base_model = backbone_policy.merge_and_unload()
        lm = PeftModel.from_pretrained(base_model, str(peft_dir), is_trainable=False)
    finally:
        peft_model_module.infer_device = original_infer_device
    lm.to(device).eval()
    return tokenizer, qwen_processor, lm


def build_detector(dabdetr_dir: Path, det_base_id: str, device: torch.device):
    try:
        det_processor = AutoImageProcessor.from_pretrained(str(dabdetr_dir), trust_remote_code=True, use_fast=True)
    except Exception:
        det_processor = AutoImageProcessor.from_pretrained(det_base_id, trust_remote_code=True, use_fast=True)

    det_model = DabDetrForObjectDetection.from_pretrained(
        str(dabdetr_dir),
        dtype=torch.float32,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    det_model.to(device).eval()
    return det_processor, det_model


def write_yolo_labels(
    label_path: Path,
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    labels: Optional[torch.Tensor],
    image_width: int,
    image_height: int,
    max_det: int,
) -> int:
    if boxes_xyxy.numel() == 0:
        label_path.write_text("", encoding="utf-8")
        return 0

    n = min(int(scores.numel()), int(max_det))
    with label_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            x0, y0, x1, y1 = [float(v) for v in boxes_xyxy[i].tolist()]
            x_center = (x0 + x1) / 2.0 / image_width
            y_center = (y0 + y1) / 2.0 / image_height
            width = (x1 - x0) / image_width
            height = (y1 - y0) / image_height
            class_id = int(labels[i].item()) if labels is not None and len(labels) > i else 0
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    return n


def filter_top_per_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: Optional[torch.Tensor],
    max_det: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if labels is None or len(labels) != len(scores):
        n = min(len(scores), max_det)
        return boxes[:n], scores[:n], labels[:n] if labels is not None else None

    best_idx: Dict[int, int] = {}
    for i, (score, label) in enumerate(zip(scores, labels)):
        class_id = int(label.item())
        if class_id < 0 or class_id > 25:
            continue
        if class_id not in best_idx or score > scores[best_idx[class_id]]:
            best_idx[class_id] = i
    keep = sorted(best_idx.values(), key=lambda idx: float(scores[idx]), reverse=True)[:max_det]
    if not keep:
        empty = torch.empty((0,), dtype=torch.long, device=labels.device)
        return boxes[:0], scores[:0], empty
    idx_tensor = torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
    return boxes[idx_tensor], scores[idx_tensor], labels[idx_tensor]


@torch.no_grad()
def run_detector(
    det_processor,
    det_model,
    image: Image.Image,
    det_feats: Optional[torch.Tensor],
    text_feats: Optional[torch.Tensor],
    device: torch.device,
    score_threshold: float,
    max_det: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    inputs = det_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    pixel_mask = inputs.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)

    model_dtype = pixel_values.dtype
    det_feats = det_feats.to(device=device, dtype=model_dtype) if det_feats is not None else None
    text_feats = text_feats.to(device=device, dtype=model_dtype) if text_feats is not None else None

    outputs = det_model(
        pixel_values=pixel_values,
        pixel_mask=pixel_mask,
        det_feats=det_feats,
        text_feats=text_feats,
        labels=None,
        return_dict=True,
    )
    processed = det_processor.post_process_object_detection(
        outputs,
        threshold=score_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    raw_boxes = processed["boxes"].detach().cpu()
    raw_scores = processed["scores"].detach().cpu()
    raw_labels = processed.get("labels", None)
    raw_labels = raw_labels.detach().cpu() if raw_labels is not None else None
    boxes, scores, labels = filter_top_per_class(raw_boxes, raw_scores, raw_labels, max_det=max_det)
    raw = {"boxes": raw_boxes, "scores": raw_scores}
    if raw_labels is not None:
        raw["labels"] = raw_labels
    return boxes, scores, labels, raw


def main() -> None:
    args = parse_args()
    t0 = time.time()
    score_threshold = float(args.score_threshold)

    test_json = Path(args.test_json)
    image_root = Path(args.image_root)
    peft_dir = Path(args.peft_dir)
    dabdetr_dir = Path(args.dabdetr_dir)
    output_dir = Path(args.output_dir)
    captions_out = output_dir / "captions"
    labels_out = output_dir / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    captions_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    print("[info] dataset=bioscifig_cap")
    print(f"[info] test_json={test_json}")
    print(f"[info] image_root={image_root}")
    print(f"[info] peft_dir={peft_dir}")
    print(f"[info] dabdetr_dir={dabdetr_dir}")
    print(f"[info] output_dir={output_dir}")
    print(f"[info] score_threshold={score_threshold}")
    print(f"[info] max_new_tokens={args.max_new_tokens} max_det={args.max_det}")
    print(f"[info] device={device} dtype={dtype}")

    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    all_items = load_test_list(test_json)
    items = [
        item
        for item_i, item in enumerate(all_items)
        if item_i % int(args.num_shards) == int(args.shard_index)
    ]
    print(f"[info] shard_index={args.shard_index} num_shards={args.num_shards} shard_items={len(items)} total_items={len(all_items)}")
    tokenizer, qwen_processor, lm = build_lm(args.lm_base_id, peft_dir, device=device, dtype=dtype)
    det_processor, det_model = build_detector(dabdetr_dir, args.det_base_id, device=device)

    out_images: List[Dict[str, Any]] = []
    caption_rows: List[Dict[str, Any]] = []
    vision_predictions: List[Dict[str, Any]] = []
    unfiltered_predictions: List[Dict[str, Any]] = []
    per_class_scores: Dict[str, Dict[str, Dict[str, Any]]] = {}
    missing_images: List[str] = []
    num_yolo_labels = 0
    captions_jsonl_path = output_dir / "captions.jsonl"
    captions_jsonl_path.write_text("", encoding="utf-8")

    for idx, item in enumerate(tqdm(items, desc="FigEx2 BioSci-Fig-Cap")):
        file_name = item.get("file_name") or item.get("image_path") or item.get("path")
        if not file_name:
            continue
        img_path = resolve_image_path(image_root, file_name)
        if img_path is None:
            missing_images.append(file_name)
            continue

        gt_for_fewshot = extract_gt_subcaptions_and_bboxes(item)
        provided_caps: Dict[str, str] = {}
        provided_boxes: Dict[int, List[float]] = {}
        for label in FEWSHOT_LABELS[: int(args.few_shot)]:
            gt_item = gt_for_fewshot.get(label) or {}
            caption = gt_item.get("caption")
            if isinstance(caption, str) and caption.strip():
                provided_caps[label] = clean_caption_text(caption)
            bbox = coerce_xyxy(gt_item.get("bbox_xyxy"))
            if bbox is not None:
                provided_boxes[label_to_class_id(label)] = bbox

        skip_lm_use_gt_only = int(args.few_shot) > 0 and len(provided_caps) < int(args.few_shot)
        raw_text = ""
        raw_assistant = ""
        parsed: Dict[str, str] = {}
        det_feats = None
        text_feats = None
        generation_details: Dict[str, Any] = {}
        if skip_lm_use_gt_only:
            for label, gt_item in gt_for_fewshot.items():
                caption = gt_item.get("caption")
                if isinstance(caption, str) and caption.strip():
                    parsed[label] = clean_caption_text(caption)
        else:
            prompt = build_fewshot_prompt(provided_caps) if int(args.few_shot) > 0 else PROMPT_COMPOUND
            try:
                raw_text, det_feats, text_feats, generation_details = lm.evaluate(
                    qwen_processor=qwen_processor,
                    tokenizer=tokenizer,
                    image_path=str(img_path),
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    prompt=prompt,
                    return_generation_details=True,
                )
            except TypeError as exc:
                if "return_generation_details" not in str(exc):
                    raise
                raw_text, det_feats, text_feats = lm.evaluate(
                    qwen_processor=qwen_processor,
                    tokenizer=tokenizer,
                    image_path=str(img_path),
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    prompt=prompt,
                )
                generation_details = {}
            raw_assistant = extract_assistant_segment(raw_text)
            parsed = parse_subcaptions(raw_text)
            if int(args.few_shot) > 0:
                parsed = {**provided_caps, **{k: v for k, v in parsed.items() if k not in provided_caps}}

        image = Image.open(img_path).convert("RGB")
        boxes, scores, labels, raw_detector = run_detector(
            det_processor=det_processor,
            det_model=det_model,
            image=image,
            det_feats=det_feats,
            text_feats=text_feats,
            device=device,
            score_threshold=score_threshold,
            max_det=int(args.max_det),
        )
        if not skip_lm_use_gt_only and int(args.few_shot) > 0:
            boxes, scores, labels = inject_provided_boxes(boxes, scores, labels, provided_boxes, max_det=int(args.max_det))
        if args.save_unfiltered_detections:
            raw_boxes = raw_detector["boxes"]
            raw_scores = raw_detector["scores"]
            raw_labels = raw_detector.get("labels")
            raw_detections = []
            for raw_i in range(int(raw_scores.numel())):
                raw_class_id = int(raw_labels[raw_i].item()) if raw_labels is not None and len(raw_labels) > raw_i else 0
                raw_detections.append(
                    {
                        "bbox_xyxy": [float(v) for v in raw_boxes[raw_i].tolist()],
                        "score": float(raw_scores[raw_i].item()),
                        "label": raw_class_id,
                    }
                )
            unfiltered_predictions.append(
                {
                    "file_name": file_name,
                    "width": image.width,
                    "height": image.height,
                    "num_after_threshold_before_top_per_class": len(raw_detections),
                    "detections": raw_detections,
                }
            )

        label_path = labels_out / f"{Path(file_name).stem}.txt"
        num_yolo_labels += write_yolo_labels(
            label_path,
            boxes,
            scores,
            labels,
            image_width=image.width,
            image_height=image.height,
            max_det=int(args.max_det),
        )

        detections: List[Dict[str, Any]] = []
        score_item: Dict[str, Dict[str, Any]] = {}
        for i in range(int(scores.numel())):
            class_id = int(labels[i].item()) if labels is not None and len(labels) > i else 0
            box = [float(v) for v in boxes[i].tolist()]
            score = float(scores[i].item())
            detections.append({"bbox_xyxy": box, "score": score, "label": class_id})
            score_item[str(class_id)] = {"score": score, "bbox_xyxy": box}
        per_class_scores[file_name] = score_item
        vision_predictions.append(
            {
                "file_name": file_name,
                "width": image.width,
                "height": image.height,
                "detections": detections,
            }
        )

        merged = dict(item)
        merged["qwen_subcaptions_raw"] = raw_assistant
        merged["qwen_subcaptions_parsed"] = parsed
        merged["detections"] = detections
        out_images.append(merged)
        caption_row = {"file_name": file_name, "raw": raw_assistant, "parsed": parsed}
        caption_rows.append(caption_row)
        caption_text = "\n".join(f"{label}: {caption}" for label, caption in sorted(parsed.items()))
        if caption_text:
            caption_text += "\n"
        (captions_out / f"{Path(file_name).stem}.txt").write_text(caption_text, encoding="utf-8")
        with captions_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(caption_row, ensure_ascii=False) + "\n")

        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            write_json({"processed": idx + 1, "elapsed_seconds": time.time() - t0}, output_dir / "progress.json")

    predictions = {"images": out_images}
    write_json(predictions, output_dir / "predictions.json")
    with captions_jsonl_path.open("w", encoding="utf-8") as f:
        for row in caption_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(per_class_scores, output_dir / "per_class_scores.json")
    write_json({"images": vision_predictions}, output_dir / "detections_with_scores.json")
    if args.save_unfiltered_detections:
        write_json({"images": unfiltered_predictions}, output_dir / "detections_unfiltered_top_per_class.json")
    summary = {
        "dataset": "bioscifig_cap",
        "test_json": str(test_json),
        "image_root": str(image_root),
        "peft_dir": str(peft_dir),
        "dabdetr_dir": str(dabdetr_dir),
        "output_dir": str(output_dir),
        "score_threshold": score_threshold,
        "max_new_tokens": int(args.max_new_tokens),
        "max_det": int(args.max_det),
        "few_shot": int(args.few_shot),
        "num_images_json": len(all_items),
        "num_images_shard": len(items),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "num_images_processed": len(out_images),
        "num_missing_images": len(missing_images),
        "num_yolo_labels": num_yolo_labels,
        "missing_images": missing_images[:50],
        "elapsed_seconds": time.time() - t0,
    }
    write_json(summary, output_dir / "run_summary.json")
    print("[done]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
