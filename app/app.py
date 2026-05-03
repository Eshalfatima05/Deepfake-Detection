"""
Step 7: Gradio Web App
- Upload image or video
- Face detection with MTCNN
- ViT inference → Real / Fake + confidence
- Attention rollout overlay visualization
"""

import sys
import os
import tempfile
from pathlib import Path

import torch
import numpy as np
import gradio as gr
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(str(Path(__file__).parent.parent))
from model.vit_model  import build_model
from model.dataset    import get_val_transforms, denormalize

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "checkpoints/vit/best_model.pth"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    print(f"Loading model from {checkpoint_path} on {DEVICE}...")
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_model(ckpt.get("model_type", "vit"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    print("Model loaded ✅")
    return model

model     = load_model(CHECKPOINT_PATH)
transform = get_val_transforms(224)

# ── MTCNN ─────────────────────────────────────────────────────────────────────
from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=DEVICE, post_process=False)


# ── Attention rollout helper ──────────────────────────────────────────────────

def attention_rollout(attentions, discard_ratio=0.9):
    result = torch.eye(attentions[0].size(-1))
    for attn in attentions:
        attn = attn.squeeze(0).mean(dim=0)
        flat = attn.view(-1)
        thr  = flat.kthvalue(int(flat.size(0) * discard_ratio)).values
        attn = torch.where(attn > thr, attn, torch.zeros_like(attn))
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = attn + torch.eye(attn.size(-1))
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        result = torch.matmul(attn, result)
    mask = result[0, 1:]
    w    = int(mask.size(0) ** 0.5)
    mask = mask.reshape(w, w)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.numpy()


def make_overlay(face_pil, attn_map):
    """Blend attention heatmap onto face image."""
    orig = np.array(face_pil.resize((224, 224))).astype(np.float32) / 255.0
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    ) / 255.0
    heatmap = cm.jet(attn_resized)[..., :3]
    overlay = np.clip(0.55 * orig + 0.45 * heatmap, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


# ── Core inference ────────────────────────────────────────────────────────────

def predict_face(face_pil: Image.Image):
    """Run model on a PIL face image. Returns label, confidence, overlay."""
    tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, attentions = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    prob_fake = probs[1].item()
    prob_real = probs[0].item()
    label     = "🔴 FAKE" if prob_fake >= 0.5 else "🟢 REAL"
    confidence = prob_fake if prob_fake >= 0.5 else prob_real

    attn_map = attention_rollout(attentions)
    overlay  = make_overlay(face_pil, attn_map)

    return label, confidence, overlay, attn_map


def detect_and_crop_face(pil_img: Image.Image):
    """Use MTCNN to detect and return cropped face. Falls back to center crop."""
    try:
        face_tensor = mtcnn(pil_img)
        if face_tensor is not None:
            face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            return Image.fromarray(face_np)
    except Exception:
        pass
    # Fallback: center crop
    w, h   = pil_img.size
    side   = min(w, h)
    left   = (w - side) // 2
    top    = (h - side) // 2
    return pil_img.crop((left, top, left + side, top + side)).resize((224, 224))


# ── Gradio handlers ───────────────────────────────────────────────────────────

def process_image(img):
    if img is None:
        return None, "No image provided.", None

    pil_img  = Image.fromarray(img).convert("RGB")
    face_pil = detect_and_crop_face(pil_img)
    label, conf, overlay, _ = predict_face(face_pil)

    conf_str = f"{conf * 100:.1f}%"
    result   = f"**Prediction:** {label}\n\n**Confidence:** {conf_str}"
    return face_pil, result, overlay


def process_video(video_path):
    if video_path is None:
        return None, "No video provided.", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video.", None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(video_fps))   # 1 fps

    frames_to_analyze = []
    frame_idx = 0
    MAX_FRAMES = 30

    while len(frames_to_analyze) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_to_analyze.append(rgb)
        frame_idx += 1
    cap.release()

    if not frames_to_analyze:
        return None, "No frames extracted.", None

    fake_probs = []
    best_face  = None
    best_overlay = None

    for frame_rgb in frames_to_analyze:
        pil_img  = Image.fromarray(frame_rgb)
        face_pil = detect_and_crop_face(pil_img)
        try:
            label, conf, overlay, _ = predict_face(face_pil)
            prob = conf if "FAKE" in label else 1 - conf
            fake_probs.append(prob)
            if best_face is None:
                best_face    = face_pil
                best_overlay = overlay
        except Exception:
            continue

    if not fake_probs:
        return None, "No faces detected in video.", None

    avg_fake_prob = float(np.mean(fake_probs))
    avg_real_prob = 1 - avg_fake_prob

    label = "🔴 FAKE" if avg_fake_prob >= 0.5 else "🟢 REAL"
    conf  = avg_fake_prob if avg_fake_prob >= 0.5 else avg_real_prob

    result = (
        f"**Prediction:** {label}\n\n"
        f"**Confidence:** {conf * 100:.1f}%\n\n"
        f"**Frames analyzed:** {len(fake_probs)}\n\n"
        f"**Avg fake probability:** {avg_fake_prob * 100:.1f}%"
    )

    return best_face, result, best_overlay


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: #666; }
.result-box { font-size: 1.1em; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="indigo")) as demo:

    gr.Markdown("# 🕵️ Deepfake Detector", elem_id="title")
    gr.Markdown(
        "Upload an **image** or **video** to detect whether it is real or AI-generated. "
        "Powered by a fine-tuned **Vision Transformer (ViT-B/16)**.",
        elem_id="subtitle"
    )

    with gr.Tabs():

        # ── Image Tab ─────────────────────────────────────────────────────────
        with gr.Tab("🖼️ Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Image", type="numpy")
                    img_btn   = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    img_face    = gr.Image(label="Detected Face", type="pil")
                    img_result  = gr.Markdown(label="Result", elem_classes="result-box")
                    img_overlay = gr.Image(label="Attention Map Overlay", type="pil")

            img_btn.click(
                fn=process_image,
                inputs=[img_input],
                outputs=[img_face, img_result, img_overlay],
            )

            gr.Examples(
                examples=[],
                inputs=[img_input],
                label="Example Images"
            )

        # ── Video Tab ─────────────────────────────────────────────────────────
        with gr.Tab("🎬 Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Upload Video (mp4/avi)")
                    vid_btn   = gr.Button("Analyze Video", variant="primary")
                with gr.Column(scale=1):
                    vid_face    = gr.Image(label="Sample Face Frame", type="pil")
                    vid_result  = gr.Markdown(label="Result", elem_classes="result-box")
                    vid_overlay = gr.Image(label="Attention Map (sample frame)", type="pil")

            vid_btn.click(
                fn=process_video,
                inputs=[vid_input],
                outputs=[vid_face, vid_result, vid_overlay],
            )

    gr.Markdown(
        "---\n"
        "**CS-419 Deep Learning Project** | "
        "Eshal Fatima (474658) · Emaan Khuram Afroze (481852) | "
        "Model: ViT-B/16 fine-tuned on FaceForensics++"
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
