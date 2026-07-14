"""
Weapon Detection System — Backend Server (Upload-Based)

FastAPI + Ultralytics YOLOv8 backend supporting image and video upload,
inference, annotated-output generation, analytics, and downloads.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.25

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("weapon-detection-api")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model: Optional[YOLO] = None
_model_load_error: Optional[str] = None

try:
    _model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded from %s", MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    _model_load_error = str(exc)
    logger.error("Failed to load YOLO model from %s: %s", MODEL_PATH, exc)


def get_model() -> YOLO:
    """Return the loaded YOLO model or raise a clear HTTP error."""
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded. Reason: {_model_load_error}",
        )
    return _model


# ---------------------------------------------------------------------------
# ffmpeg availability
# ---------------------------------------------------------------------------
# Browsers (Chrome/Edge) only decode H.264/VP9/AV1 inside <video>. OpenCV's
# VideoWriter with the "mp4v" fourcc produces MPEG-4 Part 2, which Windows
# Media Player and OpenCV itself can play (via system/FFmpeg codecs) but
# Chromium-based browsers cannot. We write the annotated frames with OpenCV
# as before, then re-encode that file to H.264 with ffmpeg before handing it
# back to the frontend.
FFMPEG_BINARY = shutil.which("ffmpeg")

if FFMPEG_BINARY is None:
    logger.error(
        "ffmpeg was not found on PATH. Video re-encoding to browser-compatible "
        "H.264 will fail until ffmpeg is installed."
    )
else:
    logger.info("ffmpeg found at %s", FFMPEG_BINARY)


def get_ffmpeg() -> str:
    """Return the ffmpeg binary path or raise a clear HTTP error."""
    if FFMPEG_BINARY is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ffmpeg is not installed or not on PATH. Install it "
                "(e.g. 'apt install ffmpeg' on Linux, 'choco install ffmpeg' "
                "or the official build on Windows, 'brew install ffmpeg' on macOS) "
                "and restart the server."
            ),
        )
    return FFMPEG_BINARY


# ---------------------------------------------------------------------------
# In-memory results store
# ---------------------------------------------------------------------------


@dataclass
class ResultRecord:
    result_id: str
    media_type: str  # "image" or "video"
    original_filename: str
    original_path: Path
    result_path: Path
    weapon_detected: bool
    weapon_name: Optional[str]
    confidence_score: Optional[float]
    objects_detected: int
    total_detections: int
    average_confidence: float
    processing_time_ms: float
    frame_count: Optional[int] = None
    class_breakdown: Dict[str, int] = field(default_factory=dict)


RESULTS: Dict[str, ResultRecord] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Weapon Detection API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded originals and generated results as static files so the
# frontend can render them directly via the returned URLs.
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def validate_extension(filename: Optional[str], allowed: set[str]) -> str:
    """Validate a filename's extension against an allowed set. Returns the lowercase extension."""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(allowed)}",
        )
    return ext


def save_upload_file(upload_file: UploadFile, dest_dir: Path, ext: str) -> Path:
    """Persist an UploadFile to disk under a UUID-based filename. Returns the saved path."""
    filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = dest_dir / filename
    try:
        with dest_path.open("wb") as buffer:
            while chunk := upload_file.file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}") from exc
    finally:
        upload_file.file.close()
    return dest_path


def extract_detections(result: Any, names: Dict[int, str]) -> List[Dict[str, Any]]:
    """Extract a list of detection dicts (class_name, confidence, bbox) from a single YOLO result."""
    detections: List[Dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        bbox = [round(c, 1) for c in boxes.xyxy[i].tolist()]
        detections.append(
            {
                "class_name": names.get(class_id, f"class_{class_id}"),
                "confidence": round(confidence, 4),
                "bbox": bbox,
            }
        )
    return detections


def summarize_detections(all_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a flat list of detections (across one image or all video frames) into summary stats."""
    total = len(all_detections)
    if total == 0:
        return {
            "weapon_detected": False,
            "weapon_name": None,
            "confidence_score": None,
            "total_detections": 0,
            "average_confidence": 0.0,
            "class_breakdown": {},
        }

    class_breakdown: Dict[str, int] = {}
    for det in all_detections:
        class_breakdown[det["class_name"]] = class_breakdown.get(det["class_name"], 0) + 1

    top_detection = max(all_detections, key=lambda d: d["confidence"])
    average_confidence = sum(d["confidence"] for d in all_detections) / total

    return {
        "weapon_detected": True,
        "weapon_name": top_detection["class_name"],
        "confidence_score": round(top_detection["confidence"], 4),
        "total_detections": total,
        "average_confidence": round(average_confidence, 4),
        "class_breakdown": class_breakdown,
    }


def run_image_inference(model: YOLO, image_path: Path, result_path: Path) -> Dict[str, Any]:
    """Run YOLO inference on a single image, save the annotated image, and return summary stats."""
    start = time.perf_counter()
    results = model(str(image_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
    processing_time_ms = (time.perf_counter() - start) * 1000.0

    if not results:
        raise HTTPException(status_code=500, detail="Model returned no results for image.")

    result = results[0]
    names = model.names

    annotated_frame = result.plot()
    success = cv2.imwrite(str(result_path), annotated_frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to write annotated image to disk.")

    detections = extract_detections(result, names)
    summary = summarize_detections(detections)
    summary["processing_time_ms"] = round(processing_time_ms, 1)
    return summary


def reencode_to_browser_h264(raw_path: Path, final_path: Path) -> None:
    """
    Re-encode a video to H.264/yuv420p with a front-loaded moov atom so it is
    playable in Chrome/Edge/Firefox <video> elements (progressive download).

    OpenCV's VideoWriter (mp4v fourcc) produces MPEG-4 Part 2, which browsers
    do not ship a decoder for — this step is what actually fixes playback.

    Notes on why this is written the way it is:
    - yuv420p REQUIRES even width and height. If the source has an odd
      dimension (very common after any resize/crop upstream), libx264 throws
      and the whole re-encode fails. The "pad" filter below forces both
      dimensions up to the nearest even number, so this can never happen.
    - "high" profile (not "baseline") is used: baseline is overly restrictive
      about resolution/level combinations and offers no real compatibility
      benefit over "high" in any modern browser.
    """
    ffmpeg_binary = get_ffmpeg()

    command = [
        ffmpeg_binary,
        "-y",  # overwrite output without prompting
        "-i", str(raw_path),
        "-an",  # no audio track was written by OpenCV, so don't try to map one
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        str(final_path),
    ]

    logger.info("Running ffmpeg re-encode: %s", " ".join(command))

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Always log ffmpeg's stderr (it logs progress/info there even on success)
    # so you have full visibility in the server console when debugging.
    if completed.stderr:
        logger.info("ffmpeg output:\n%s", completed.stderr[-4000:])

    if completed.returncode != 0 or not final_path.exists():
        logger.error("ffmpeg re-encode failed with return code %s", completed.returncode)
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg re-encode to browser-compatible H.264 failed: {completed.stderr[-2000:]}",
        )


def safe_unlink(path: Path, attempts: int = 5, delay_seconds: float = 0.3) -> None:
    """
    Delete a file, retrying briefly if it's still locked.

    On Windows, a file can remain briefly locked immediately after a process
    (OpenCV's VideoWriter or ffmpeg) closes its handle. Retrying avoids a
    spurious failure on an otherwise-successful request.
    """
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == attempts - 1:
                logger.warning("Could not delete temp file %s after %d attempts", path, attempts)
                return
            time.sleep(delay_seconds)


def run_video_inference(model: YOLO, video_path: Path, result_path: Path) -> Dict[str, Any]:
    """Run YOLO inference on every frame of a video, write a browser-playable annotated output video, and return stats."""
    # Fail fast if ffmpeg isn't available, before spending time on inference.
    get_ffmpeg()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open uploaded video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Uploaded video has invalid dimensions.")

    # Write the raw annotated frames to a throwaway temp file first (mp4v is
    # fine here — it's an intermediate, never served to the browser).
    raw_path = result_path.with_name(f"{result_path.stem}_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Failed to initialize video writer.")

    names = model.names
    all_detections: List[Dict[str, Any]] = []
    frame_count = 0

    start = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            result = results[0]
            annotated_frame = result.plot()
            writer.write(annotated_frame)

            all_detections.extend(extract_detections(result, names))
    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        safe_unlink(raw_path)
        raise HTTPException(status_code=400, detail="No frames could be read from the uploaded video.")

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail="OpenCV wrote an empty video file. This usually means the installed "
            "OpenCV build has no working MP4/mp4v encoder backend.",
        )

    try:
        reencode_to_browser_h264(raw_path, result_path)
    finally:
        safe_unlink(raw_path)

    processing_time_ms = (time.perf_counter() - start) * 1000.0

    summary = summarize_detections(all_detections)
    summary["processing_time_ms"] = round(processing_time_ms, 1)
    summary["frame_count"] = frame_count
    return summary


def generate_live_frames():
    """
    Capture webcam frames, run YOLO inference and continuously
    stream annotated JPEG frames to the frontend.
    """
    model = get_model()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Unable to open webcam.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
            annotated_frame = frame.copy()

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0]) * 100

                label = f"Weapon Detected"
                #label = f"Weapon {confidence:.0f}%"

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            success, buffer = cv2.imencode(".jpg", annotated_frame)

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


def build_media_urls(original_path: Path, result_path: Path) -> Dict[str, str]:
    return {
        "originalUrl": f"/uploads/{original_path.name}",
        "resultUrl": f"/results/{result_path.name}",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    healthy = _model is not None and FFMPEG_BINARY is not None
    return {
        "status": "healthy" if healthy else "degraded",
        "model_path": MODEL_PATH,
        "model_loaded": _model is not None,
        "model_load_error": _model_load_error,
        "ffmpeg_available": FFMPEG_BINARY is not None,
        "ffmpeg_path": FFMPEG_BINARY,
    }


@app.get("/video_feed")
async def video_feed():
    """Live webcam stream with YOLO detections."""
    return StreamingResponse(
        generate_live_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    model = get_model()
    ext = validate_extension(file.filename, IMAGE_EXTENSIONS)

    original_path = save_upload_file(file, UPLOAD_DIR, ext)
    result_id = uuid.uuid4().hex
    result_path = RESULTS_DIR / f"{result_id}.jpg"

    try:
        summary = run_image_inference(model, original_path, result_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Image inference failed")
        raise HTTPException(status_code=500, detail=f"Image inference failed: {exc}") from exc

    urls = build_media_urls(original_path, result_path)

    record = ResultRecord(
        result_id=result_id,
        media_type="image",
        original_filename=file.filename or original_path.name,
        original_path=original_path,
        result_path=result_path,
        weapon_detected=summary["weapon_detected"],
        weapon_name=summary["weapon_name"],
        confidence_score=summary["confidence_score"],
        objects_detected=summary["total_detections"],
        total_detections=summary["total_detections"],
        average_confidence=summary["average_confidence"],
        processing_time_ms=summary["processing_time_ms"],
        class_breakdown=summary["class_breakdown"],
    )
    RESULTS[result_id] = record

    return {
        "success": True,
        "resultId": result_id,
        "weaponDetected": record.weapon_detected,
        "weaponName": record.weapon_name,
        "confidenceScore": record.confidence_score,
        "objectsDetected": record.objects_detected,
        "resultImageUrl": urls["resultUrl"],
        "originalImageUrl": urls["originalUrl"],
        "analytics": {
            "totalDetections": record.total_detections,
            "averageConfidence": record.average_confidence,
            "processingTimeMs": record.processing_time_ms,
            "classBreakdown": record.class_breakdown,
        },
    }


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    model = get_model()
    ext = validate_extension(file.filename, VIDEO_EXTENSIONS)

    original_path = save_upload_file(file, UPLOAD_DIR, ext)
    result_id = uuid.uuid4().hex
    result_path = RESULTS_DIR / f"{result_id}.mp4"

    try:
        summary = run_video_inference(model, original_path, result_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Video inference failed")
        raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc

    urls = build_media_urls(original_path, result_path)

    record = ResultRecord(
        result_id=result_id,
        media_type="video",
        original_filename=file.filename or original_path.name,
        original_path=original_path,
        result_path=result_path,
        weapon_detected=summary["weapon_detected"],
        weapon_name=summary["weapon_name"],
        confidence_score=summary["confidence_score"],
        objects_detected=summary["total_detections"],
        total_detections=summary["total_detections"],
        average_confidence=summary["average_confidence"],
        processing_time_ms=summary["processing_time_ms"],
        frame_count=summary["frame_count"],
        class_breakdown=summary["class_breakdown"],
    )
    RESULTS[result_id] = record

    return {
        "success": True,
        "resultId": result_id,
        "weaponDetected": record.weapon_detected,
        "weaponName": record.weapon_name,
        "confidenceScore": record.confidence_score,
        "objectsDetected": record.objects_detected,
        "resultVideoUrl": urls["resultUrl"],
        "originalVideoUrl": urls["originalUrl"],
        "downloadUrl": f"/api/download/{result_id}",
        "analytics": {
            "totalDetections": record.total_detections,
            "averageConfidence": record.average_confidence,
            "processingTimeMs": record.processing_time_ms,
            "frameCount": record.frame_count,
            "classBreakdown": record.class_breakdown,
        },
    }


@app.get("/api/results/{result_id}")
async def get_result(result_id: str) -> Dict[str, Any]:
    record = RESULTS.get(result_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found.")

    urls = build_media_urls(record.original_path, record.result_path)

    return {
        "success": True,
        "resultId": record.result_id,
        "mediaType": record.media_type,
        "originalFilename": record.original_filename,
        "weaponDetected": record.weapon_detected,
        "weaponName": record.weapon_name,
        "confidenceScore": record.confidence_score,
        "objectsDetected": record.objects_detected,
        "originalUrl": urls["originalUrl"],
        "resultUrl": urls["resultUrl"],
        "downloadUrl": f"/api/download/{record.result_id}",
        "analytics": {
            "totalDetections": record.total_detections,
            "averageConfidence": record.average_confidence,
            "processingTimeMs": record.processing_time_ms,
            "frameCount": record.frame_count,
            "classBreakdown": record.class_breakdown,
        },
    }


@app.get("/api/download/{result_id}")
async def download_result(result_id: str) -> FileResponse:
    record = RESULTS.get(result_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found.")

    if not record.result_path.exists():
        raise HTTPException(status_code=404, detail="Result file no longer exists on disk.")

    media_type = "image/jpeg" if record.media_type == "image" else "video/mp4"
    download_filename = f"{record.result_id}{record.result_path.suffix}"

    return FileResponse(
        path=str(record.result_path),
        media_type=media_type,
        filename=download_filename,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)