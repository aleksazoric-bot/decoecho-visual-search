import json
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model (free, local)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load your products
with open("products.json", "r") as f:
    products = json.load(f)

embeddings = []
meta = []

def fetch_image(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None

for p in products:
    img_url = p.get("image")
    if not img_url:
        continue

    img = fetch_image(img_url)
    if img is None:
        continue

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)   # normalize
    emb = emb.cpu().numpy()[0]

    embeddings.append(emb)
    meta.append({
        "handle": p["handle"],
        "title": p.get("title", p["handle"]),
        "image": img_url
    })

# Save
np.save("product_embeddings.npy", np.vstack(embeddings))
with open("product_meta.json", "w") as f:
    json.dump(meta, f)

print(f"Saved {len(meta)} product embeddings.")
