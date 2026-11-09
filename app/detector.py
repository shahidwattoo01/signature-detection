import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
from PIL import Image
import io
import os

CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")

_model = None
_device = None


def get_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    model.to(_device)
    model.eval()

    _model = model
    print(f"Model loaded on {_device}")
    return _model, _device


def detect_signatures(image_bytes: bytes) -> list:
    model, device = get_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    img_resized = img.resize((416, 416))
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    scale_x = orig_w / 416
    scale_y = orig_h / 416

    results = []
    for box, score in zip(predictions["boxes"], predictions["scores"]):
        if score.item() >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box.cpu().numpy()
            results.append({
                "x1": float(x1 * scale_x),
                "y1": float(y1 * scale_y),
                "x2": float(x2 * scale_x),
                "y2": float(y2 * scale_y),
                "confidence": float(score.item())
            })

    return results
