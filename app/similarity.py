import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io
import numpy as np

_extractor = None
_device = None


def get_extractor():
    global _extractor, _device
    if _extractor is not None:
        return _extractor, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = resnet18(weights=ResNet18_Weights.DEFAULT)
    _extractor = torch.nn.Sequential(*list(base.children())[:-1])
    _extractor.to(_device)
    _extractor.eval()
    return _extractor, _device


transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def _get_embedding(image_bytes: bytes) -> np.ndarray:
    extractor, device = get_extractor()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = extractor(tensor).squeeze().cpu().numpy()
    return embedding


def compute_similarity(image1_bytes: bytes, image2_bytes: bytes) -> float:
    emb1 = _get_embedding(image1_bytes)
    emb2 = _get_embedding(image2_bytes)

    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    similarity = float(dot / (norm + 1e-8))

    similarity = max(0.0, min(1.0, similarity))
    return round(similarity, 4)
