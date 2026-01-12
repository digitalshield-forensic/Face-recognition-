#!/usr/bin/env python3
"""
face_mvp.py

Minimal single-user face recognition MVP using Google Cloud Vision for face detection
and landmark feature extraction. Known-face encodings are computed from face landmarks
and attributes and stored locally. New images or webcam frames are sent to Vision, encoded,
and compared to known encodings using cosine similarity.

Requirements (install instructions below):
 - python3.8+
 - google-cloud-vision
 - opencv-python
 - pillow
 - numpy

Usage examples:
 - Build/refresh encodings from known images:
     python face_mvp.py --known-dir ./known_faces --encodings encodings.npz --credentials /path/key.json --build
 - Recognize a single image:
     python face_mvp.py --input test.jpg --encodings encodings.npz --credentials /path/key.json
 - Run webcam (press 'q' to quit, 's' to force-check current frame):
     python face_mvp.py --input webcam --encodings encodings.npz --credentials /path/key.json
"""
import os
import io
import sys
import time
import math
import json
import glob
import argparse
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import cv2

from google.cloud import vision
from google.api_core import exceptions as gcloud_exceptions

# ----- Configurable constants -----
DEFAULT_ENCODING_FILE = "encodings.npz"
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LANDMARK_ORDER = [
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_OF_LEFT_EYEBROW",
    "RIGHT_OF_LEFT_EYEBROW",
    "LEFT_OF_RIGHT_EYEBROW",
    "RIGHT_OF_RIGHT_EYEBROW",
    "MIDPOINT_BETWEEN_EYES",
    "NOSE_TIP",
    "UPPER_LIP",
    "LOWER_LIP",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_EAR_TRAGION",
    "RIGHT_EAR_TRAGION",
    "LEFT_EYE_TOP_BOUNDARY",
    "LEFT_EYE_RIGHT_CORNER",
    "LEFT_EYE_LEFT_CORNER",
    "LEFT_EYE_BOTTOM_BOUNDARY",
    "RIGHT_EYE_TOP_BOUNDARY",
    "RIGHT_EYE_RIGHT_CORNER",
    "RIGHT_EYE_LEFT_CORNER",
    "RIGHT_EYE_BOTTOM_BOUNDARY",
    # add or remove entries as desired
]
LIKELIHOOD_MAP = {
    0: 0.0,  # UNKNOWN
    1: 0.0,  # VERY_UNLIKELY
    2: 0.2,  # UNLIKELY
    3: 0.5,  # POSSIBLE
    4: 0.8,  # LIKELY
    5: 1.0,  # VERY_LIKELY
}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_mvp")


# ----- Utility helpers -----
def set_credentials_env(credentials_path: Optional[str]):
    if credentials_path:
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        logger.debug("Set GOOGLE_APPLICATION_CREDENTIALS=%s", credentials_path)


def get_vision_client() -> vision.ImageAnnotatorClient:
    try:
        client = vision.ImageAnnotatorClient()
        logger.debug("Created Google Vision client")
        return client
    except Exception as e:
        logger.exception("Failed to create Vision client: %s", e)
        raise


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def image_bytes_from_cv2(frame) -> bytes:
    # Encode to jpg in memory
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame to jpg bytes")
    return encoded.tobytes()


def detect_faces_with_retry(client: vision.ImageAnnotatorClient, image_bytes: bytes, retries=3, backoff=1.0):
    attempt = 0
    while True:
        attempt += 1
        try:
            image = vision.Image(content=image_bytes)
            response = client.face_detection(image=image)
            if response.error.message:
                raise RuntimeError(f"Vision API error: {response.error.message}")
            return response.face_annotations
        except gcloud_exceptions.GoogleAPICallError as e:
            logger.warning("Google API call error on attempt %d: %s", attempt, e)
            if attempt >= retries:
                logger.exception("Exceeded retries for Vision API call")
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))
        except Exception as e:
            logger.exception("Unexpected error during face detection: %s", e)
            raise


def bbox_from_face(face) -> Tuple[int, int, int, int]:
    # bounding_poly vertices are relative to image coordinates (integers)
    xs = [v.x for v in face.bounding_poly.vertices if hasattr(v, "x") and v.x is not None]
    ys = [v.y for v in face.bounding_poly.vertices if hasattr(v, "y") and v.y is not None]
    if not xs or not ys:
        # fall back to landmarks if bounding box missing (rare)
        xs = [lm.position.x for lm in face.landmarks if hasattr(lm.position, "x")]
        ys = [lm.position.y for lm in face.landmarks if hasattr(lm.position, "y")]
        if not xs or not ys:
            return 0, 0, 0, 0
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return int(xmin), int(ymin), int(xmax), int(ymax)


def landmark_dict(face) -> Dict[str, Tuple[float, float, float]]:
    d = {}
    for lm in face.landmarks:
        # lm.type is enum int, map to name
        try:
            name = vision.FaceAnnotation.Landmark.Type.Name(lm.type)
        except Exception:
            # fallback: use numeric type
            name = f"TYPE_{lm.type}"
        pos = lm.position
        d[name] = (getattr(pos, "x", 0.0), getattr(pos, "y", 0.0), getattr(pos, "z", 0.0))
    return d


def compute_encoding_from_face(face) -> np.ndarray:
    """
    Build a fixed-length encoding vector from:
      - normalized landmark (x,y) positions for a chosen ordered list LANDMARK_ORDER
      - normalized bbox size (w,h)
      - detection_confidence
      - emotion likelihoods (joy, sorrow, anger, surprise, headwear)
    The final vector is L2-normalized.
    """
    landmarks = landmark_dict(face)
    xmin, ymin, xmax, ymax = bbox_from_face(face)
    w = float(max(1.0, xmax - xmin))
    h = float(max(1.0, ymax - ymin))

    vals = []

    # landmarks (normalize within bbox)
    for lname in LANDMARK_ORDER:
        if lname in landmarks:
            x, y, z = landmarks[lname]
            nx = (x - xmin) / w
            ny = (y - ymin) / h
            vals.extend([nx, ny])
        else:
            # missing -> zeros
            vals.extend([0.0, 0.0])

    # bbox size normalized by image area is not available here (we don't have image dims),
    # but we can add relative bbox w/h (w and h will be rescaled below).
    vals.append(w)
    vals.append(h)

    # detection confidence
    detection_conf = getattr(face, "detection_confidence", 0.0)
    vals.append(float(detection_conf))

    # Likelihoods converted to numeric (joy, sorrow, anger, surprise, headwear)
    try:
        vals.append(float(LIKELIHOOD_MAP.get(face.joy_likelihood, 0.0)))
        vals.append(float(LIKELIHOOD_MAP.get(face.sorrow_likelihood, 0.0)))
        vals.append(float(LIKELIHOOD_MAP.get(face.anger_likelihood, 0.0)))
        vals.append(float(LIKELIHOOD_MAP.get(face.surprise_likelihood, 0.0)))
        vals.append(float(LIKELIHOOD_MAP.get(face.headwear_likelihood, 0.0)))
    except Exception:
        vals.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    vec = np.array(vals, dtype=np.float32)
    # simple normalization: scale down bbox w/h to ratio by dividing by (w+h) to reduce scale sensitivity
    # and L2 normalize
    if np.linalg.norm(vec) > 1e-6:
        vec = vec / np.linalg.norm(vec)
    else:
        vec = vec
    return vec


def save_encodings(path: str, encodings: List[np.ndarray], labels: List[str], meta: Optional[dict] = None):
    # Save as compressed numpy file: stack encodings to 2D array if same shape
    if not encodings:
        raise ValueError("No encodings to save")
    stack = np.stack(encodings, axis=0)
    meta = meta or {}
    np.savez_compressed(path, encodings=stack, labels=np.array(labels), meta=json.dumps(meta))
    logger.info("Saved %d encodings to %s", len(labels), path)


def load_encodings(path: str) -> Tuple[List[np.ndarray], List[str], dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encodings file not found: {path}")
    data = np.load(path, allow_pickle=True)
    enc = [row.astype(np.float32) for row in data["encodings"]]
    labels = [str(x) for x in data["labels"].tolist()]
    meta = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"].tolist()))
        except Exception:
            meta = {}
    logger.info("Loaded %d encodings from %s", len(labels), path)
    return enc, labels, meta


def build_known_encodings(client: vision.ImageAnnotatorClient, known_dir: str, retries=3) -> Tuple[List[np.ndarray], List[str]]:
    image_paths = []
    for ext in SUPPORTED_IMAGE_EXTS:
        image_paths.extend(glob.glob(os.path.join(known_dir, f"*{ext}")))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No images found in known-dir: {known_dir}")

    encodings = []
    labels = []
    for path in image_paths:
        label = os.path.splitext(os.path.basename(path))[0]
        try:
            b = read_file_bytes(path)
            faces = detect_faces_with_retry(client, b, retries=retries)
            if not faces:
                logger.warning("No face found in known image %s; skipping", path)
                continue
            # Take first face only for the label
            face = faces[0]
            vec = compute_encoding_from_face(face)
            encodings.append(vec)
            labels.append(label)
            logger.info("Processed known face '%s' from %s", label, path)
        except Exception as e:
            logger.exception("Failed to process known image %s: %s", path, e)
    return encodings, labels


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return -1.0
    # ensure float32
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def match_encoding(known_encodings: List[np.ndarray], known_labels: List[str], probe: np.ndarray) -> Tuple[Optional[str], float]:
    best_score = -1.0
    best_label = None
    for kvec, klabel in zip(known_encodings, known_labels):
        try:
            score = cosine_similarity(kvec, probe)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_label = klabel
    return best_label, best_score


def draw_annotation_on_image_cv2(image_bgr, face, label_text: str):
    xmin, ymin, xmax, ymax = bbox_from_face(face)
    # clip
    h, w = image_bgr.shape[:2]
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w - 1, xmax), min(h - 1, ymax)
    cv2.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # text background
    cv2.rectangle(image_bgr, (xmin, ymin - 20), (xmin + 250, ymin), (0, 255, 0), -1)
    cv2.putText(image_bgr, label_text[:240], (xmin + 2, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# ----- Main execution flow -----
def main():
    parser = argparse.ArgumentParser(description="MVP face recognition using Google Cloud Vision")
    parser.add_argument("--known-dir", help="Directory with known face images (one person per image, filename=label)", default=None)
    parser.add_argument("--encodings", help="Path to encodings file", default=DEFAULT_ENCODING_FILE)
    parser.add_argument("--credentials", help="Path to Google service account JSON (sets GOOGLE_APPLICATION_CREDENTIALS)", default=None)
    parser.add_argument("--input", help="Input image path or 'webcam' for live feed", default=None)
    parser.add_argument("--threshold", help="Cosine similarity threshold for match (0..1)", type=float, default=0.7)
    parser.add_argument("--build", help="(Re)build known encodings from --known-dir", action="store_true")
    parser.add_argument("--no-visual", help="Do not show OpenCV GUI (useful for headless)", action="store_true")
    parser.add_argument("--frame-skip", help="For webcam: only send every N frames to Vision to reduce cost/latency", type=int, default=30)
    parser.add_argument("--max-retries", help="API call retries", type=int, default=3)
    args = parser.parse_args()

    if not args.encodings and not args.known_dir:
        parser.error("Either --encodings or --known-dir must be provided (use --build to create encodings)")

    # credentials
    try:
        set_credentials_env(args.credentials)
    except Exception as e:
        logger.error("Credential error: %s", e)
        sys.exit(2)

    # create client
    try:
        client = get_vision_client()
    except Exception:
        logger.error("Failed to initialize Google Vision client")
        sys.exit(2)

    # Build or load encodings
    known_encodings = []
    known_labels = []
    if args.build:
        if not args.known_dir:
            logger.error("--build requires --known-dir")
            sys.exit(2)
        try:
            encs, labels = build_known_encodings(client, args.known_dir, retries=args.max_retries)
            if not encs:
                logger.error("No encodings built from known-dir")
                sys.exit(2)
            save_encodings(args.encodings, encs, labels, meta={"landmark_order": LANDMARK_ORDER})
            known_encodings, known_labels = encs, labels
        except Exception as e:
            logger.exception("Failed to build known encodings: %s", e)
            sys.exit(2)
    else:
        # load encodings if available
        if os.path.exists(args.encodings):
            try:
                encs, labels, meta = load_encodings(args.encodings)
                known_encodings, known_labels = encs, labels
            except Exception as e:
                logger.exception("Failed to load encodings: %s", e)
                sys.exit(2)
        else:
            if args.known_dir:
                # build implicitly if encodings file not found
                logger.info("Encodings file not found; building from known-dir")
                try:
                    encs, labels = build_known_encodings(client, args.known_dir, retries=args.max_retries)
                    save_encodings(args.encodings, encs, labels, meta={"landmark_order": LANDMARK_ORDER})
                    known_encodings, known_labels = encs, labels
                except Exception as e:
                    logger.exception("Failed to build known encodings: %s", e)
                    sys.exit(2)
            else:
                logger.error("Encodings file not found and no known-dir provided")
                sys.exit(2)

    # if input is not provided, exit after building/loading encodings
    if not args.input:
        logger.info("Encodings ready at %s. Provide --input <image|webcam> to run recognition.", args.encodings)
        return

    # Single image input recognition
    if args.input.lower() != "webcam":
        if not os.path.exists(args.input):
            logger.error("Input image not found: %s", args.input)
            sys.exit(2)
        try:
            b = read_file_bytes(args.input)
            faces = detect_faces_with_retry(client, b, retries=args.max_retries)
            # load image for drawing
            pil = Image.open(io.BytesIO(b)).convert("RGB")
            image_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            if not faces:
                logger.warning("No faces found in input image")
                if not args.no_visual:
                    cv2.imshow("result", image_cv)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                return
            for face in faces:
                probe = compute_encoding_from_face(face)
                label, score = match_encoding(known_encodings, known_labels, probe)
                if score >= args.threshold:
                    text = f"{label} ({score:.2f})"
                else:
                    text = f"Unknown ({score:.2f})"
                draw_annotation_on_image_cv2(image_cv, face, text)
                logger.info("Face match: %s (score %.4f)", label, score)
            if not args.no_visual:
                cv2.imshow("result", image_cv)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            logger.exception("Failed to process input image: %s", e)
            sys.exit(2)
        return

    # Webcam mode
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        sys.exit(2)
    frame_count = 0
    last_processed_time = 0.0
    logger.info("Starting webcam. Press 'q' to quit, 's' to force-check current frame.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                break
            display_frame = frame.copy()
            frame_count += 1
            processed = False
            should_process = (frame_count % args.frame_skip == 0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                should_process = True
            if should_process:
                try:
                    logger.debug("Sending frame to Vision API (this may be slow)...")
                    image_bytes = image_bytes_from_cv2(frame)
                    faces = detect_faces_with_retry(client, image_bytes, retries=args.max_retries)
                    for face in faces:
                        probe = compute_encoding_from_face(face)
                        label, score = match_encoding(known_encodings, known_labels, probe)
                        if score >= args.threshold:
                            text = f"{label} ({score:.2f})"
                        else:
                            text = f"Unknown ({score:.2f})"
                        draw_annotation_on_image_cv2(display_frame, face, text)
                        logger.info("Webcam face match: %s (score %.4f)", label, score)
                    last_processed_time = time.time()
                    processed = True
                except Exception as e:
                    logger.exception("Error processing webcam frame: %s", e)
            # show info overlay
            overlay_text = f"Frame {frame_count} | Processed: {processed} | LastProc: {time.strftime('%H:%M:%S', time.localtime(last_processed_time))}"
            cv2.putText(display_frame, overlay_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if not args.no_visual:
                cv2.imshow("webcam", display_frame)
    finally:
        cap.release()
        if not args.no_visual:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()