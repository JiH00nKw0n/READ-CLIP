from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Mayfull/READ-CLIP"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img_url = "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg"
response = requests.get(img_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

caption = "A man with a red helmet on a small moped on a dirt road."

inputs = processor(
    text=[caption],
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image

print(f"Similarity score: {logits_per_image.item():.4f}")
