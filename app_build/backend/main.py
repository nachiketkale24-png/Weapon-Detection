"""
Real-Time Weapon Tracking App — Backend Server
================================================
FastAPI application with a WebSocket endpoint that:
  1. Receives base64-encoded JPEG video frames from the frontend.
  2. Decodes them into OpenCV-compatible NumPy arrays.
  3. Runs YOLOv8 inference with ByteTrack persistent tracking.
  4. Returns bounding boxes, tracking IDs, confidences, and class names as JSON.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import asyncio
import base64
import json
import time
import logging
from contextlib import asynccontextmanager
from functools import partial

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the YOLOv8 model weights.
# Defaults to "best.pt" (teammate's trained model) in the current directory.
# Override with the MODEL_PATH environment variable if needed.
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")

# Confidence threshold for YOLOv8 inference (server-side minimum).
# Detections below this threshold are discarded before sending to the client.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("weapon-tracker")

# ---------------------------------------------------------------------------
# Model Singleton
# ---------------------------------------------------------------------------

model: YOLO | None = None

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the YOLOv8 model on startup, cleanup on shutdown."""
    global model
    logger.info("Loading YOLOv8 model from: %s", MODEL_PATH)
    try:
        model = YOLO(MODEL_PATH)
        logger.info(
            "✅ Model loaded successfully — classes: %s",
            list(model.names.values()) if model.names else "N/A",
        )
    except Exception as exc:
        logger.error("❌ Failed to load model: %s", exc)
        raise RuntimeError(f"Could not load YOLOv8 model from '{MODEL_PATH}'") from exc

    yield  # Application runs here

    # Shutdown cleanup
    logger.info("🛑 Shutting down — releasing model resources.")
    model = None

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Weapon Tracking API",
    description="Real-time weapon detection & tracking via WebSocket + YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Streamlit and Vite dev servers during local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit default dev server
        "http://localhost:5173",   # Vite default dev server
        "http://localhost:3000",   # Alternate React dev port
        "http://127.0.0.1:8501",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


# ---------------------------------------------------------------------------
# Utility: Decode a base64-encoded JPEG into an OpenCV image (ndarray)
# ---------------------------------------------------------------------------

def decode_base64_frame(base64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded JPEG string into an OpenCV BGR image.

    Pipeline:
        base64 string → raw bytes → numpy buffer → cv2.imdecode (BGR ndarray)

    Raises:
        ValueError: If the base64 string is malformed or the image cannot be decoded.
    """
    try:
        # Strip optional data-URI prefix (e.g., "data:image/jpeg;base64,...")
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_string)
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("cv2.imdecode returned None — invalid image data.")

        return frame

    except Exception as exc:
        raise ValueError(f"Failed to decode base64 frame: {exc}") from exc


# ---------------------------------------------------------------------------
# Utility: Run YOLOv8 tracking and extract structured detections
# ---------------------------------------------------------------------------

def run_inference_sync(frame: np.ndarray, frame_count: int) -> dict:
    """
    Run YOLOv8 object tracking on a single frame and return structured results.

    Uses `.track(persist=True, tracker='bytetrack.yaml')` for persistent
    multi-object tracking across sequential frames.

    NOTE: This function is synchronous (CPU/GPU-bound). It is called via
    asyncio.to_thread() in the WebSocket handler to avoid blocking the event loop.

    Args:
        frame:       BGR image as a NumPy ndarray.
        frame_count: Monotonically increasing frame counter.

    Returns:
        dict with keys: detections, inference_time_ms, frame_count
    """
    start_time = time.perf_counter()

    # Run tracking with ByteTrack persistence
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,  # Suppress per-frame console output
    )

    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    detections = []

    if results and len(results) > 0:
        result = results[0]  # Single image → single result

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            # --- CUSTOM CLASS DICTIONARY ADDED HERE ---
            # Edit these names to match exactly what your 6 classes are
            MY_CLASSES = {
                0: "knife",
                1: "gun",
                2: "sword",
                3: "rifle",
                4: "pistol",
                5: "unknown_weapon"
            }
            # ------------------------------------------

            for i in range(len(boxes)):
                # Bounding box coordinates (xyxy format)
                bbox = boxes.xyxy[i].tolist()
                # Round to 1 decimal place for cleaner JSON
                bbox = [round(coord, 1) for coord in bbox]

                # Confidence score
                confidence = float(boxes.conf[i])

                # Class ID → class name
                class_id = int(boxes.cls[i])
                
                # --- CHANGED LINE HERE TO USE CUSTOM DICTIONARY ---
                class_name = MY_CLASSES.get(class_id, f"class_{class_id}")
                # --------------------------------------------------

                # Tracking ID (may be None if tracker hasn't assigned one yet)
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i])

                detections.append({
                    "bbox": bbox,
                    "track_id": track_id,
                    "confidence": round(confidence, 3),
                    "class_name": class_name,
                })

    return {
        "detections": detections,
        "inference_time_ms": round(inference_time_ms, 1),
        "frame_count": frame_count,
    }


# ---------------------------------------------------------------------------
# Core WebSocket Handler Logic
# ---------------------------------------------------------------------------

async def _handle_tracking_session(websocket: WebSocket):
    """
    Core WebSocket handler — shared by /ws and /ws/track routes.

    Protocol:
        Client → Server:  { "frame": "<base64-JPEG>" }
        Server → Client:  { "detections": [...], "inference_time_ms": float, "frame_count": int }
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("🔌 WebSocket connected: %s", client_host)

    frame_count = 0

    try:
        while True:
            # ------ Receive frame from client ------
            raw_message = await websocket.receive_text()

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format."})
                continue

            base64_frame = message.get("frame")
            if not base64_frame:
                await websocket.send_json({"error": "Missing 'frame' field in message."})
                continue

            # ------ Decode the frame ------
            try:
                frame = decode_base64_frame(base64_frame)
            except ValueError as exc:
                logger.warning("⚠️  Frame decode error: %s", exc)
                await websocket.send_json({"error": str(exc)})
                continue

            # ------ Run YOLOv8 inference in a thread (non-blocking) ------
            frame_count += 1
            response = await asyncio.to_thread(run_inference_sync, frame, frame_count)

            # ------ Send results back to client ------
            await websocket.send_json(response)

            # Log periodically (every 50 frames) to avoid log flooding
            if frame_count % 50 == 0:
                det_count = len(response["detections"])
                logger.info(
                    "📊 Frame #%d | %d detections | %.1fms inference",
                    frame_count,
                    det_count,
                    response["inference_time_ms"],
                )

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected: %s (after %d frames)", client_host, frame_count)

    except Exception as exc:
        logger.error("❌ WebSocket error: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass  # Connection may already be closed


# ---------------------------------------------------------------------------
# WebSocket Endpoints: /ws and /ws/track (separate functions)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint at /ws for real-time weapon detection and tracking."""
    await _handle_tracking_session(websocket)


@app.websocket("/ws/track")
async def websocket_track_endpoint(websocket: WebSocket):
    """WebSocket endpoint at /ws/track for real-time weapon detection and tracking."""
    await _handle_tracking_session(websocket)


# ---------------------------------------------------------------------------
# Entry Point (for direct execution: `python main.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )