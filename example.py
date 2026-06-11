from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "Mayfull/READ-CLIP"
PROCESSOR_NAME = "openai/clip-vit-base-patch32"
IMAGE_URL = "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg"

CAPTIONS = [
    (
        "positive",
        "A man with a red helmet is riding a small moped on a dirt road.",
    ),
    (
        "negative",
        "A small moped with a red helmet is riding a man on a dirt road.",
    ),
]


def load_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(PROCESSOR_NAME)

    image = load_image(IMAGE_URL)
    labels, texts = zip(*CAPTIONS)

    inputs = processor(
        text=list(texts),
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    scores = outputs.logits_per_image[0].detach().cpu()
    ranking = sorted(zip(labels, texts, scores.tolist()), key=lambda item: item[2], reverse=True)

    print(f"Device: {device}")
    print(f"Image: {IMAGE_URL}\n")
    print("Ranked captions:")
    for rank, (label, text, score) in enumerate(ranking, start=1):
        print(f"{rank}. {label:<8} | score={score:.4f}")
        print(f"   {text}")


if __name__ == "__main__":
    main()
