import io
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import uvicorn
import json

# ----- Config -----
TOP_K = 12  # how many results to return

# ----- App setup -----
app = FastAPI()

# Allow your Shopify domain to call this API
origins = [
    "https://decoecho.com",
    "https://www.decoecho.com",
    # add your preview domain if needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Load model + embeddings -----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

embeddings = np.load("product_embeddings.npy")  # shape: (N, 512)
with open("product_meta.json", "r") as f:
    meta = json.load(f)

embeddings_torch = torch.from_numpy(embeddings).to(device)

def compute_embedding(image: Image.Image):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
    return img_emb

def search_similar(img_emb, top_k=TOP_K):
    # cosine similarity with precomputed embeddings
    scores = (img_emb @ embeddings_torch.T).squeeze(0)  # (N,)
    topk = torch.topk(scores, k=min(top_k, scores.shape[0]))
    indices = topk.indices.cpu().numpy().tolist()
    values = topk.values.cpu().numpy().tolist()

    results = []
    for idx, score in zip(indices, values):
        p = meta[idx]
        results.append({
            "handle": p["handle"],
            "title": p["title"],
            "image": p["image"],
            "score": float(score)
        })
    return results

# ----- Routes -----
@app.post("/visual-search")
async def visual_search(image: UploadFile = File(...)):
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return {"error": "Invalid image"}

    img_emb = compute_embedding(img)
    results = search_similar(img_emb)

    return {"products": results}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

