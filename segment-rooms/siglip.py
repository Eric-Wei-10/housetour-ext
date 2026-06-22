import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

HF_TOKEN = os.environ.get("HF_TOKEN")

SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
SIGLIP_CACHE_DIR  = "/cluster/scratch/xinwei/model_checkpoints"

HOUSE_ROOM_TYPES = [
    "living room",
    "bedroom",
    "bathroom",
    "kitchen",
    "dining room",
    "hallway",
    "office",
    "study",
    "garage",
    "laundry room",
    "storage room",
    "balcony",
    "garden",
    "outside",
]

_TEMPLATES = [
    "{}",
    "There is the {} in the scene.",
]


def load_siglip_model(device: str):
    print(f"Loading {SIGLIP_MODEL_NAME} from transformers ...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME, cache_dir=SIGLIP_CACHE_DIR, token=HF_TOKEN)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, cache_dir=SIGLIP_CACHE_DIR, token=HF_TOKEN)
    model.eval().to(device)
    return model, processor


def extract_siglip_features(image_paths, model_and_processor, device: str, batch_size: int = 32):
    """Return float32 ndarray (N, D), L2-normalised SigLIP vision CLS embeddings."""
    model, processor = model_and_processor
    all_feats = []
    batch = []

    def _flush(imgs):
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output
        return F.normalize(feats.float(), dim=-1).cpu().numpy()

    for path in tqdm(image_paths, desc="  SigLIP", leave=False):
        batch.append(Image.open(path).convert("RGB"))
        if len(batch) == batch_size:
            all_feats.append(_flush(batch))
            batch = []
    if batch:
        all_feats.append(_flush(batch))

    return np.concatenate(all_feats, axis=0).astype(np.float32)


extract_features = extract_siglip_features


def build_room_text_embeddings(model_and_processor, device: str):
    """Return (text_embs, logit_scale, logit_bias).

    text_embs:   float32 (C, D)  L2-normalised, one embedding per room category
    logit_scale: float scalar    learned temperature (exp applied)
    logit_bias:  float scalar    learned additive bias
    """
    model, processor = model_and_processor
    prompts = [t.format(cat) for cat in HOUSE_ROOM_TYPES for t in _TEMPLATES]

    inputs = processor(text=prompts, return_tensors="pt",
                       padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        text_feats   = model.get_text_features(**inputs)  # (C * num_templates, D)
        if not isinstance(text_feats, torch.Tensor):
            text_feats = text_feats.pooler_output
        logit_scale  = model.logit_scale.exp().item()
        logit_bias   = model.logit_bias.item()
    text_feats = F.normalize(text_feats.float(), dim=-1)

    n_templates = len(_TEMPLATES)
    text_feats = text_feats.view(len(HOUSE_ROOM_TYPES), n_templates, -1)
    text_feats = F.normalize(text_feats.mean(dim=1), dim=-1)  # (C, D)

    return text_feats.cpu().numpy().astype(np.float32), logit_scale, logit_bias


def classify_rooms(siglip_feats: np.ndarray, text_embeddings: np.ndarray,
                   logit_scale: float = 1.0, logit_bias: float = 0.0):
    """Classify each frame into a room category.

    Args:
        siglip_feats:    float32 (N, D) L2-normalised image embeddings
        text_embeddings: float32 (C, D) L2-normalised text embeddings
        logit_scale:     learned temperature factor
        logit_bias:      learned additive bias

    Returns:
        labels: int32   (N,)    index into HOUSE_ROOM_TYPES
        scores: float32 (N, C)  per-category sigmoid probabilities
    """
    logits = siglip_feats @ text_embeddings.T * logit_scale + logit_bias  # (N, C)
    scores = 1.0 / (1.0 + np.exp(-logits))                                # sigmoid
    labels = scores.argmax(axis=1).astype(np.int32)
    return labels, scores.astype(np.float32)
