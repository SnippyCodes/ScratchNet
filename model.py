import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 7 flower categories trained by the model
CATEGORIES: List[str] = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
    "bellflower",
    "lotus",
]

# Detailed botanical and aesthetic metadata for each flower class
FLOWER_DETAILS: Dict[str, Dict[str, Any]] = {
    "daisy": {
        "scientific_name": "Bellis perennis",
        "family": "Asteraceae",
        "icon": "🌼",
        "color": "#facc15",
        "tagline": "Radiant Innocence & Simplicity",
        "description": "Daisies feature classic white ray florets surrounding a bright yellow center disc. Known for closing their petals at night and reopening with the morning light.",
        "season": "Spring through Autumn",
    },
    "dandelion": {
        "scientific_name": "Taraxacum officinale",
        "family": "Asteraceae",
        "icon": "🦁",
        "color": "#eab308",
        "tagline": "Golden Resilience & Wish-Maker",
        "description": "Known for its vivid golden yellow flower heads that mature into delicate spherical puffballs of downy seeds carried on the breeze.",
        "season": "Spring through Summer",
    },
    "rose": {
        "scientific_name": "Rosa",
        "family": "Rosaceae",
        "icon": "🌹",
        "color": "#f43f5e",
        "tagline": "Timeless Elegance & Romance",
        "description": "Celebrated worldwide for velvety spiral petals, sweet intoxicating fragrance, and thorny canes. Universal symbol of passion, love, and grace.",
        "season": "Late Spring through Autumn",
    },
    "sunflower": {
        "scientific_name": "Helianthus annuus",
        "family": "Asteraceae",
        "icon": "🌻",
        "color": "#f59e0b",
        "tagline": "Solar Majesty & Warmth",
        "description": "Famous for heliotropism (tracking the path of the sun). Features towering stalks topped by magnificent large heads with vibrant yellow petals.",
        "season": "Mid Summer through Early Autumn",
    },
    "tulip": {
        "scientific_name": "Tulipa",
        "family": "Liliaceae",
        "icon": "🌷",
        "color": "#ec4899",
        "tagline": "Springtime Vibrance & Renewal",
        "description": "Elegant cup-shaped blooms in symmetrical proportions. Highly prized throughout history, notably sparking the famous Dutch Golden Age tulip boom.",
        "season": "Early to Mid Spring",
    },
    "bellflower": {
        "scientific_name": "Campanula",
        "family": "Campanulaceae",
        "icon": "🔔",
        "color": "#8b5cf6",
        "tagline": "Whimsical Violet Grace",
        "description": "Distinguished by delightful bell-like five-lobed petals blooming in rich shades of purple, blue, violet, and occasionally white.",
        "season": "Late Spring to Mid Summer",
    },
    "lotus": {
        "scientific_name": "Nelumbo nucifera",
        "family": "Nelumbonaceae",
        "icon": "🪷",
        "color": "#06b6d4",
        "tagline": "Sacred Purity & Enlightenment",
        "description": "An aquatic perennial rooted in muddy riverbeds that rises pristine above water. Revered in ancient cultures as a sacred symbol of spiritual rebirth and resilience.",
        "season": "Summer",
    },
}

TARGET_SIZE: Tuple[int, int] = (150, 150)


class FlowerClassifier(nn.Module):
    """
    Convolutional Neural Network (CNN) architecture trained on the 7-flower dataset.
    
    Structure:
      - 3 Convolutional Blocks: (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
      - Flatten: 128 channels * 18 * 18 spatial resolution = 41,472 features
      - Fully Connected Dense Layers: 41,472 -> 128 -> 64 -> 32 -> 7 logits
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        # Conv Block 1: 3 -> 32 channels (150x150 -> 75x75)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 2: 32 -> 64 channels (75x75 -> 37x37)
        self.cnn_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 3: 64 -> 128 channels (37x37 -> 18x18)
        self.cnn_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dense Classifier Head
        self.flatten = nn.Flatten()
        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=128 * 18 * 18, out_features=128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(),
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.ReLU(),
        )
        self.out = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layer(x)
        x = self.cnn_layer_2(x)
        x = self.cnn_layer_3(x)
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.out(x)
        return x


_cached_model: Union[FlowerClassifier, None] = None
_cached_device: Union[torch.device, None] = None


def get_device() -> torch.device:
    global _cached_device
    if _cached_device is None:
        _cached_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _cached_device


def load_model(weights_path: Union[str, Path] = "flower_classifier.pth") -> Tuple[FlowerClassifier, torch.device]:
    """
    Loads the trained PyTorch model with cached singleton pattern.
    """
    global _cached_model
    device = get_device()

    if _cached_model is not None:
        return _cached_model, device

    path = Path(weights_path)
    if not path.is_file():
        # Check current directory or parent
        alt_path = Path(__file__).resolve().parent / weights_path
        if alt_path.is_file():
            path = alt_path
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    model = FlowerClassifier(num_classes=len(CATEGORIES)).to(device)
    state_dict = torch.load(str(path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _cached_model = model
    return _cached_model, device


def preprocess_image(image_input: Union[Image.Image, bytes, bytearray, str, Path]) -> torch.Tensor:
    """
    Preprocess an image exactly as done during model training:
      1. Open / convert to RGB format.
      2. Resize to (150, 150).
      3. Scale pixel values from [0, 255] to [0.0, 1.0].
      4. Convert to float32 tensor with shape (1, 3, 150, 150).
    """
    if isinstance(image_input, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize to model target input size
    image = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    
    # Array to (H, W, C) float32 normalized
    arr = np.array(image, dtype=np.float32) / 255.0
    
    # Permute to (C, H, W) and add batch dim -> (1, 3, 150, 150)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict_flower(image_input: Union[Image.Image, bytes, bytearray, str, Path]) -> Dict[str, Any]:
    """
    Runs model inference on the provided flower image.
    Returns:
      - 'top_flower': Predicted class name (e.g. 'sunflower')
      - 'confidence': Prediction probability (0.0 to 1.0)
      - 'confidence_pct': Formatted percentage string (e.g. '98.5%')
      - 'probabilities': Dictionary of {category: probability} sorted descending
      - 'ranked_list': List of {name, probability, percentage, color, icon}
      - 'flower_info': Botanical metadata dictionary
      - 'device': Inference hardware device ('cuda' or 'cpu')
    """
    model, device = load_model()
    tensor = preprocess_image(image_input).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    sorted_indices = np.argsort(probs)[::-1]
    top_idx = int(sorted_indices[0])
    top_category = CATEGORIES[top_idx]
    top_confidence = float(probs[top_idx])

    ranked_list = []
    probabilities_dict = {}
    for idx in sorted_indices:
        cat = CATEGORIES[idx]
        p = float(probs[idx])
        probabilities_dict[cat] = p
        detail = FLOWER_DETAILS.get(cat, {})
        ranked_list.append({
            "name": cat,
            "display_name": cat.capitalize(),
            "probability": round(p, 4),
            "percentage": f"{p * 100:.1f}%",
            "percent_value": round(p * 100, 1),
            "color": detail.get("color", "#10b981"),
            "icon": detail.get("icon", "🌸"),
        })

    return {
        "top_flower": top_category,
        "display_name": top_category.capitalize(),
        "confidence": round(top_confidence, 4),
        "confidence_pct": f"{top_confidence * 100:.1f}%",
        "probabilities": probabilities_dict,
        "ranked_list": ranked_list,
        "flower_info": FLOWER_DETAILS.get(top_category, {}),
        "device": str(device),
    }
