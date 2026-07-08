"""
Real-Time Weapon Tracking App — Backend Server
"""

import os
import asyncio
import base64
import json
import time
import logging
from typing import Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("weapon-tracker")

app = FastAPI(title="Weapon Tracking API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-Memory Pub/Sub State
# ---------------------------------------------------------------------------
# sessions[session_id] = {"viewer_ws": WebSocket, "producer_connected": bool}
active_sessions: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_path": MODEL_PATH}

@app.get("/api/cameras")
async def get_cameras():
    """Scan local cameras using pygrabber (Windows only)"""
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        return [{"index": idx, "name": name} for idx, name in enumerate(devices)]
    except Exception as e:
        logger.warning(f"Could not scan cameras via pygrabber: {e}")
        return []

@app.get("/phone", response_class=HTMLResponse)
async def phone_camera_page(session_id: str):
    """Serve a lightweight webpage to stream the phone's camera."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Virtual Camera Producer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            body {{ background: #000; color: #fff; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }}
            video {{ width: 100%; max-width: 600px; border-radius: 10px; margin-top: 20px; }}
            .status {{ margin-top: 10px; padding: 10px; background: #333; border-radius: 5px; }}
            #stopBtn {{ margin-top: 20px; padding: 10px 20px; background: #e94560; border: none; border-radius: 5px; color: white; font-size: 16px; display: none; }}
        </style>
    </head>
    <body>
        <h2>Live Phone Camera</h2>
        <div class="status" id="status">Connecting...</div>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <button id="stopBtn" onclick="stopStream()">Stop Streaming</button>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const status = document.getElementById('status');
            const stopBtn = document.getElementById('stopBtn');
            const ctx = canvas.getContext('2d');
            let ws;
            let stream;
            let intervalId;

            async function startStream() {{
                try {{
                    stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "environment" }}, audio: false }});
                    video.srcObject = stream;
                    status.innerText = "Camera Access Granted. Connecting to Server...";
                    
                    // Determine WS protocol
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${{protocol}}//${{window.location.host}}/ws/produce/{session_id}`;
                    ws = new WebSocket(wsUrl);

                    let isProcessing = false;
                    
                    ws.onmessage = (event) => {{
                        isProcessing = false;
                    }};

                    ws.onopen = () => {{
                        status.innerText = "Connected to Dashboard! Streaming...";
                        stopBtn.style.display = 'inline-block';
                        
                        // Send frames at up to 10 FPS, waiting for backend ACK
                        intervalId = setInterval(() => {{
                            if (isProcessing) return; // wait for server
                            if (video.videoWidth > 0 && ws.readyState === WebSocket.OPEN) {{
                                isProcessing = true;
                                const scale = Math.min(1.0, 640 / video.videoWidth);
                                canvas.width = video.videoWidth * scale;
                                canvas.height = video.videoHeight * scale;
                                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                                const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
                                ws.send(JSON.stringify({{ frame: dataUrl }}));
                            }}
                        }}, 100);
                    }};

                    ws.onclose = () => {{
                        status.innerText = "Disconnected from Dashboard.";
                        stopStream();
                    }};
                    
                    ws.onerror = (e) => {{
                        status.innerText = "WebSocket Error!";
                    }};

                }} catch (err) {{
                    status.innerText = "Error accessing camera: " + err.message;
                }}
            }}

            function stopStream() {{
                if (intervalId) clearInterval(intervalId);
                if (ws && ws.readyState === WebSocket.OPEN) ws.close();
                if (stream) stream.getTracks().forEach(t => t.stop());
                video.srcObject = null;
                status.innerText = "Stream Stopped.";
                stopBtn.style.display = 'none';
            }}

            window.onload = startStream;
        </script>
    </body>
    </html>
    """

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def decode_base64_frame(base64_string: str) -> np.ndarray:
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_string)
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image data.")
    return frame

def run_inference_sync(model: YOLO, frame: np.ndarray, frame_count: int) -> dict:
    start_time = time.perf_counter()
    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", conf=CONFIDENCE_THRESHOLD, verbose=False)
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            MY_CLASSES = {0: "knife", 1: "gun", 2: "sword", 3: "rifle", 4: "pistol", 5: "unknown_weapon"}
            for i in range(len(boxes)):
                bbox = [round(c, 1) for c in boxes.xyxy[i].tolist()]
                conf = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = MY_CLASSES.get(class_id, f"class_{class_id}")
                track_id = int(boxes.id[i]) if boxes.id is not None else None

                detections.append({
                    "bbox": bbox,
                    "track_id": track_id,
                    "confidence": round(conf, 3),
                    "class_name": class_name,
                })

    return {
        "detections": detections,
        "inference_time_ms": round(inference_time_ms, 1),
        "frame_count": frame_count,
    }


# ---------------------------------------------------------------------------
# WebSockets
# ---------------------------------------------------------------------------

@app.websocket("/ws/track")
async def websocket_track_endpoint(websocket: WebSocket):
    """Legacy/Local camera tracking where the client sends frames and receives JSON."""
    await websocket.accept()
    
    try:
        session_model = YOLO(MODEL_PATH)
    except Exception as exc:
        await websocket.close(code=1011)
        return

    frame_count = 0
    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            base64_frame = message.get("frame")
            if not base64_frame: continue
            frame = decode_base64_frame(base64_frame)
            frame_count += 1
            response = await asyncio.to_thread(run_inference_sync, session_model, frame, frame_count)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws/view/{session_id}")
async def websocket_view_endpoint(websocket: WebSocket, session_id: str):
    """Streamlit connects here to VIEW a virtual phone camera."""
    await websocket.accept()
    active_sessions[session_id] = {"viewer_ws": websocket, "producer_connected": False}
    try:
        # Keep connection open and wait for producer
        while True:
            # We don't expect messages from the viewer, but we ping to keepalive
            await asyncio.sleep(1)
            # If producer disconnected, we could notify
    except WebSocketDisconnect:
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        logger.error(f"Error in ws/view: {e}", exc_info=True)
        if session_id in active_sessions:
            del active_sessions[session_id]


@app.websocket("/ws/produce/{session_id}")
async def websocket_produce_endpoint(websocket: WebSocket, session_id: str):
    """Phone connects here to SEND frames."""
    await websocket.accept()
    
    # Wait for viewer to be ready instead of immediately rejecting
    wait_time = 0
    while session_id not in active_sessions and wait_time < 60:
        await asyncio.sleep(1)
        wait_time += 1
        
    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="No viewer ready for this session.")
        return
        
    viewer_ws = active_sessions[session_id]["viewer_ws"]
    active_sessions[session_id]["producer_connected"] = True
    
    try:
        session_model = YOLO(MODEL_PATH)
    except Exception:
        await websocket.close(code=1011)
        return

    frame_count = 0
    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            base64_frame = message.get("frame")
            if not base64_frame: continue
            
            frame = decode_base64_frame(base64_frame)
            frame_count += 1
            response = await asyncio.to_thread(run_inference_sync, session_model, frame, frame_count)
            
            # Forward the original base64 frame AND the detections to the Viewer (Streamlit)
            # Streamlit needs the frame to draw on it
            viewer_message = {
                "frame": base64_frame,
                "detections": response["detections"],
                "inference_time_ms": response["inference_time_ms"],
                "frame_count": response["frame_count"]
            }
            try:
                await viewer_ws.send_json(viewer_message)
                # Send ACK to the phone so it knows it can send the next frame
                await websocket.send_json({"status": "ok"})
            except Exception:
                # Viewer might have disconnected
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error in ws/produce: {e}", exc_info=True)
    finally:
        active_sessions[session_id]["producer_connected"] = False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)