import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
from flask import Flask, request, jsonify
import cv2, numpy as np, base64
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

app = Flask(__name__)

# Force CPU
device = torch.device('cpu')
torch.set_num_threads(max(1, torch.get_num_threads()))  # keep CPU threads sane

# MTCNN for detection + alignment; returns a cropped/aligned face tensor
mtcnn = MTCNN(keep_all=False, device=device, image_size=160, post_process=True)

# InceptionResnetV1 (VGGFace2) for embeddings
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def decode_image(b64: str):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

def crop_bbox(img_bgr: np.ndarray, box):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]

@app.post("/detect")
def detect():
    try:
        data = request.json or {}
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "missing 'image' base64"}), 400

        # 1) Decode input
        img_bgr = decode_image(img_b64)
        if img_bgr is None:
            return jsonify({"error": "invalid image"}), 400

        # Convert to PIL RGB for MTCNN
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 2) Detect bbox for returning cropped face preview
        boxes, probs = mtcnn.detect(img_pil)
        if boxes is None or len(boxes) == 0 or probs[0] is None or probs[0] < 0.90:
            return jsonify({"error": "face not detected"}), 400

        # Take the most confident face
        idx = int(np.argmax(probs))
        box = boxes[idx]  # [x1, y1, x2, y2]

        # 3) Get aligned face tensor for embedding
        #    MTCNN(img) returns aligned tensor (C,H,W) in [-1,1]
        face_tensor = mtcnn(img_pil)
        if face_tensor is None:
            return jsonify({"error": "face not detected"}), 400

        # 4) Embedding (L2-normalized)
        with torch.no_grad():
            emb = embedder(face_tensor.unsqueeze(0).to(device))  # (1,512)
            emb = F.normalize(emb, p=2, dim=1)
        embedding = emb.squeeze(0).cpu().numpy().astype(np.float32).tolist()

        # 5) Crop the face to return as image preview
        face_img = crop_bbox(img_bgr, box)
        if face_img is None:
            return jsonify({"error": "failed to crop face"}), 500

        ok, buffer = cv2.imencode(".jpg", face_img)
        if not ok:
            return jsonify({"error": "failed to encode face image"}), 500
        face_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "face_image": face_b64,
            "embedding": embedding
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/compare")
def compare():
    try:
        payload = request.json or {}
        e1 = np.array(payload.get("e1"), dtype=np.float32)
        e2 = np.array(payload.get("e2"), dtype=np.float32)

        if e1.size == 0 or e2.size == 0:
            return jsonify({"error": "missing 'e1' or 'e2'"}), 400

        # Cosine similarity
        denom = (np.linalg.norm(e1) * np.linalg.norm(e2)) + 1e-10
        sim = float(np.dot(e1, e2) / denom)

        # Threshold can be tuned; 0.6 is a reasonable starting point for Facenet
        return jsonify({"match": sim > 0.6, "similarity": sim})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use host="0.0.0.0" if you want external access
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000,debug=False)