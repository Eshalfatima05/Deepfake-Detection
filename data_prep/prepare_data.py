"""
Step 2: Data Preparation
- Extracts frames from FaceForensics++ videos at 1fps
- Detects and crops faces using MTCNN
- Organizes into real/ and fake/ folders
- Creates train/val/test splits
"""

import os
import cv2
import json
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ── MTCNN for face detection ──────────────────────────────────────────────────
from facenet_pytorch import MTCNN
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "ff_root":        "data/FaceForensics",          # root of your FF++ download
    "output_dir":     "data/processed",              # where cropped faces go
    "image_size":     224,
    "fps_extract":    1,                             # extract 1 frame per second
    "max_frames":     150,                           # max frames per video
    "train_ratio":    0.80,
    "val_ratio":      0.10,
    "test_ratio":     0.10,
    "min_face_conf":  0.95,                          # MTCNN confidence threshold
    "seed":           42,
    # FF++ manipulation types to include
    "fake_methods": [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "FaceShifter",
    ],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── MTCNN Setup ───────────────────────────────────────────────────────────────
mtcnn = MTCNN(
    image_size=CONFIG["image_size"],
    margin=20,
    min_face_size=60,
    thresholds=[0.6, 0.7, 0.9],
    factor=0.709,
    keep_all=False,         # keep only the largest face
    device=DEVICE,
    post_process=False,     # return 0-255 uint8
)


def extract_frames(video_path: str, fps: int = 1, max_frames: int = 150):
    """Extract frames from video at given fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0
    frame_interval = max(1, int(round(video_fps / fps)))

    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # BGR → RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()
    return frames


def crop_face(frame_rgb: np.ndarray):
    """
    Run MTCNN on a single frame.
    Returns PIL Image (224x224) or None if no face found.
    """
    pil_img = Image.fromarray(frame_rgb)
    try:
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is None or probs is None:
            return None
        if probs[0] < CONFIG["min_face_conf"]:
            return None
        # crop using the first (most confident) box
        face_tensor = mtcnn(pil_img)  # returns (3, H, W) float tensor or None
        if face_tensor is None:
            return None
        face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(face_np)
    except Exception:
        return None


def process_video(video_path: str, out_dir: str, label: str, video_id: str):
    """Extract frames → crop faces → save to out_dir/label/video_id_frame.jpg"""
    frames = extract_frames(
        video_path,
        fps=CONFIG["fps_extract"],
        max_frames=CONFIG["max_frames"],
    )
    saved = 0
    for i, frame in enumerate(frames):
        face = crop_face(frame)
        if face is None:
            continue
        save_path = os.path.join(out_dir, label, f"{video_id}_f{i:04d}.jpg")
        face.save(save_path, quality=95)
        saved += 1
    return saved


def build_dataset():
    ff_root   = Path(CONFIG["ff_root"])
    out_root  = Path(CONFIG["output_dir"])

    # Create output dirs
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    # ── Collect video paths ───────────────────────────────────────────────────
    # FF++ folder structure:
    #   FaceForensics/
    #     original_sequences/youtube/c23/videos/*.mp4   ← real
    #     manipulated_sequences/<method>/c23/videos/*.mp4 ← fake

    real_videos = sorted(
        (ff_root / "original_sequences" / "youtube" / "c23" / "videos").glob("*.mp4")
    )

    fake_videos = []
    for method in CONFIG["fake_methods"]:
        method_path = ff_root / "manipulated_sequences" / method / "c23" / "videos"
        if method_path.exists():
            fake_videos += sorted(method_path.glob("*.mp4"))

    print(f"Found {len(real_videos)} real videos")
    print(f"Found {len(fake_videos)} fake videos across {len(CONFIG['fake_methods'])} methods")

    random.seed(CONFIG["seed"])

    # ── Process real videos ───────────────────────────────────────────────────
    # Temp collect all face images into a flat list then split
    temp_real = out_root / "_temp_real"
    temp_fake = out_root / "_temp_fake"
    temp_real.mkdir(parents=True, exist_ok=True)
    temp_fake.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Processing REAL videos...")
    for vpath in tqdm(real_videos):
        vid_id = vpath.stem
        process_video(str(vpath), str(temp_real.parent), "real", vid_id)
        # save directly to temp
        frames = extract_frames(str(vpath), CONFIG["fps_extract"], CONFIG["max_frames"])
        for i, frame in enumerate(frames):
            face = crop_face(frame)
            if face is None:
                continue
            face.save(str(temp_real / f"{vid_id}_f{i:04d}.jpg"), quality=95)

    print("\n[2/2] Processing FAKE videos...")
    for vpath in tqdm(fake_videos):
        vid_id = vpath.stem + "_" + vpath.parent.parent.parent.name  # add method name
        frames = extract_frames(str(vpath), CONFIG["fps_extract"], CONFIG["max_frames"])
        for i, frame in enumerate(frames):
            face = crop_face(frame)
            if face is None:
                continue
            face.save(str(temp_fake / f"{vid_id}_f{i:04d}.jpg"), quality=95)

    # ── Split and move ────────────────────────────────────────────────────────
    def split_and_move(temp_dir, label):
        all_files = list(temp_dir.glob("*.jpg"))
        random.shuffle(all_files)
        n = len(all_files)
        n_train = int(n * CONFIG["train_ratio"])
        n_val   = int(n * CONFIG["val_ratio"])

        splits = {
            "train": all_files[:n_train],
            "val":   all_files[n_train:n_train + n_val],
            "test":  all_files[n_train + n_val:],
        }
        for split, files in splits.items():
            for f in files:
                shutil.move(str(f), str(out_root / split / label / f.name))
        print(f"  {label}: train={len(splits['train'])} | val={len(splits['val'])} | test={len(splits['test'])}")

    print("\nSplitting dataset...")
    split_and_move(temp_real, "real")
    split_and_move(temp_fake, "fake")

    # cleanup temp
    shutil.rmtree(str(temp_real))
    shutil.rmtree(str(temp_fake))

    # ── Save split manifest ───────────────────────────────────────────────────
    manifest = {}
    for split in ["train", "val", "test"]:
        manifest[split] = {}
        for label in ["real", "fake"]:
            files = list((out_root / split / label).glob("*.jpg"))
            manifest[split][label] = len(files)

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Dataset ready!")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build_dataset()
