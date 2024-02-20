from pathlib import Path

import cv2
from PIL import Image
import torch

from src.dto import PredictOutput
from src.utils import create_mask_from_bbox, mask_to_pil


def predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "cpu",  # "cuda", "mps", or "cpu"
    half: bool = False,
    verbose: bool = True,
) -> PredictOutput:
    from ultralytics import YOLO

    model = YOLO(model_path)
    device = torch.device(device)
    pred = model(
        image, conf=confidence, device=device, half=half, verbose=verbose)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)
    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)
