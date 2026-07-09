"""
Real-Time Weapon Tracking App — Streamlit Frontend
"""

import streamlit as st
import cv2
import numpy as np
import base64
import json
import time
import threading
import uuid
import requests
import qrcode
from io import BytesIO
# pyrefly: ignore [missing-import]
import websocket
# pyrefly: ignore [missing-import]
from websocket._exceptions import WebSocketConnectionClosedException

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🔫 Weapon Tracker — Multi-Camera",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 { color: #e94560; font-size: 2rem; margin: 0 0 0.3rem 0; font-weight: 700; }
    .main-header p { color: #8892b0; font-size: 0.95rem; margin: 0; }
    .metrics-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(233, 69, 96, 0.15); border-radius: 10px;
        padding: 1rem 1.5rem; flex: 1; min-width: 150px; text-align: center;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card .label { color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.3rem; }
    .metric-card .value { color: #e94560; font-size: 1.6rem; font-weight: 700; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    [data-testid="stSidebar"] h2 { color: #e94560; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Color Palette & Drawing Utility
# ─────────────────────────────────────────────────────────────────────────────

BOX_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255), (255, 128, 0),
]

def get_box_color(track_id: int | None) -> tuple:
    if track_id is None:
        return (0, 0, 255)
    return BOX_COLORS[track_id % len(BOX_COLORS)]



def draw_detections(frame, detections, conf_threshold, thickness, f_scale, draw_confidence):
    drawn_count = 0
    for det in detections:
        confidence = det.get("confidence", 0.0)
        if confidence < conf_threshold: continue
        bbox = det.get("bbox", [])
        if len(bbox) != 4: continue
        x1, y1, x2, y2 = [int(c) for c in bbox]
        track_id = det.get("track_id")
        
        class_name = det.get("class_name", "unknown")
        
        color = get_box_color(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_name}"
        if draw_confidence:
            label += f" {confidence:.0%}"

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 1)
        label_y = max(y1 - 10, text_h + 5)
        cv2.rectangle(frame, (x1, label_y - text_h - 5), (x1 + text_w + 6, label_y + baseline), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 3, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (255, 255, 255), 1, cv2.LINE_AA)
        drawn_count += 1
    return frame, drawn_count

def decode_base64_frame(base64_string: str) -> np.ndarray:
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_string)
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

# ─────────────────────────────────────────────────────────────────────────────
# Background Worker Thread for Each Camera
# ─────────────────────────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    def __init__(self, mode, source, name, backend_url, conf_thresh, box_thick, font_scale, show_conf, frame_interval_ms):
        super().__init__()
        self.mode = mode # "local" or "virtual"
        self.source = source
        self.name = name
        self.backend_url = backend_url
        self.conf_thresh = conf_thresh
        self.box_thick = box_thick
        self.font_scale = font_scale
        self.show_conf = show_conf
        self.frame_interval_ms = frame_interval_ms
        
        self.running = True
        self.latest_frame = None
        self.status = "Initializing..."
        self.metrics = {"fps": 0.0, "detections": 0, "inference_ms": 0.0}
        self.error_msg = None

    def run(self):
        ws_url = self.backend_url.replace("http://", "ws://").replace("https://", "wss://")
        
        if self.mode == "local":
            self.run_local(ws_url)
        elif self.mode == "virtual":
            self.run_virtual(ws_url)

    def run_local(self, base_ws_url):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status = "Camera Error"
            self.error_msg = f"Failed to open local camera: {self.name}"
            self.running = False
            return
            
        try:
            ws = websocket.create_connection(f"{base_ws_url}/ws/track", timeout=10)
        except Exception as e:
            self.status = "WebSocket Error"
            self.error_msg = str(e)
            cap.release()
            self.running = False
            return

        self.status = "Connected"
        frame_count = 0
        prev_time = time.perf_counter()

        while self.running:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                self.status = "Camera Disconnected"
                break

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success: continue
                
            b64_frame = base64.b64encode(buffer).decode("utf-8")
            
            try:
                ws.send(json.dumps({"frame": b64_frame}))
                ws.settimeout(15)  # Increased timeout to 15s to allow YOLO model to load on first frame
                response = json.loads(ws.recv())
            except Exception as e:
                self.status = "WebSocket Error"
                self.error_msg = str(e)
                break

            if "error" in response:
                self.status = "Server Error"
                self.error_msg = response["error"]
                break

            detections = response.get("detections", [])
            annotated_frame, drawn_count = draw_detections(
                frame, detections, self.conf_thresh, self.box_thick, self.font_scale, self.show_conf
            )
            self.latest_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            frame_count += 1
            current_time = time.perf_counter()
            elapsed = current_time - prev_time
            if elapsed >= 1.0:
                self.metrics["fps"] = frame_count / elapsed
                frame_count = 0
                prev_time = current_time
                
            self.metrics["detections"] = drawn_count
            self.metrics["inference_ms"] = response.get("inference_time_ms", 0.0)
            
            elapsed_loop_ms = (time.perf_counter() - loop_start) * 1000
            sleep_ms = max(0, self.frame_interval_ms - elapsed_loop_ms)
            if sleep_ms > 0: time.sleep(sleep_ms / 1000.0)

        try: ws.close()
        except: pass
        cap.release()
        self.running = False
        
    def run_virtual(self, base_ws_url):
        # source is the session_id
        session_id = self.source
        self.status = "Waiting for Phone..."
        
        try:
            ws = websocket.create_connection(f"{base_ws_url}/ws/view/{session_id}", timeout=60)
        except Exception as e:
            self.status = "WebSocket Error"
            self.error_msg = str(e)
            self.running = False
            return

        self.status = "Connected to Phone"
        frame_count = 0
        prev_time = time.perf_counter()

        while self.running:
            try:
                # Passive receive
                ws.settimeout(2)
                raw = ws.recv()
                response = json.loads(raw)
            except websocket.WebSocketTimeoutException:
                # Keep waiting
                continue
            except Exception as e:
                self.status = "WebSocket Closed"
                break

            if "error" in response:
                self.status = "Server Error"
                self.error_msg = response["error"]
                break

            base64_frame = response.get("frame")
            detections = response.get("detections", [])
            if not base64_frame: continue
            
            try:
                frame = decode_base64_frame(base64_frame)
            except Exception:
                continue

            annotated_frame, drawn_count = draw_detections(
                frame, detections, self.conf_thresh, self.box_thick, self.font_scale, self.show_conf
            )
            self.latest_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            frame_count += 1
            current_time = time.perf_counter()
            elapsed = current_time - prev_time
            if elapsed >= 1.0:
                self.metrics["fps"] = frame_count / elapsed
                frame_count = 0
                prev_time = current_time
                
            self.metrics["detections"] = drawn_count
            self.metrics["inference_ms"] = response.get("inference_time_ms", 0.0)
            
        try: ws.close()
        except: pass
        self.running = False

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────

import socket
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

if "tracking_active" not in st.session_state:
    st.session_state.tracking_active = False
if "workers" not in st.session_state:
    st.session_state.workers = []
if "camera_list" not in st.session_state:
    st.session_state.camera_list = [{"index": 0, "name": "Default Webcam"}]
if "virtual_cams" not in st.session_state:
    st.session_state.virtual_cams = [] # List of dicts: {"session_id": "...", "name": "Phone X"}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Settings Panel
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    default_url = f"http://{get_local_ip()}:8000"
    backend_url = st.text_input("🌐 Backend URL", value=default_url)
    
    # Auto-scan cameras once on load
    if "auto_scanned" not in st.session_state:
        try:
            r = requests.get(f"{backend_url}/api/cameras", timeout=3)
            data = r.json()
            if data:
                st.session_state.camera_list = data
        except Exception:
            pass
        st.session_state.auto_scanned = True
        
    st.markdown("### 📷 Cameras")
    
    # --- Local Cameras ---
    if st.button("🔄 Scan Local Cameras"):
        try:
            r = requests.get(f"{backend_url}/api/cameras", timeout=3)
            data = r.json()
            if data:
                st.session_state.camera_list = data
                st.success(f"Found {len(data)} camera(s)!")
            else:
                st.warning("No cameras found or OS not supported.")
        except Exception as e:
            st.error(f"Failed to scan: {e}")
            
    # Create dict mapping name -> index for local cameras
    local_cam_map = {cam["name"]: cam["index"] for cam in st.session_state.camera_list}
    
    selected_local_names = st.multiselect(
        "Local Webcams",
        options=list(local_cam_map.keys()),
        default=list(local_cam_map.keys())[0] if local_cam_map else None,
    )
    
    # --- Virtual Cameras ---
    st.markdown("---")
    if st.button("📱 Add Virtual Phone Camera"):
        new_session = str(uuid.uuid4())[:8] # short uuid
        st.session_state.virtual_cams.append({
            "session_id": new_session,
            "name": f"Phone {new_session}"
        })
        st.session_state.tracking_active = False # Force worker reload
        st.rerun()

    # Display active virtual cameras & QR codes
    for vcam in st.session_state.virtual_cams:
        with st.expander(f"📷 {vcam['name']}", expanded=True):
            phone_url = f"{backend_url}/phone?session_id={vcam['session_id']}"
            qr = qrcode.make(phone_url)
            buf = BytesIO()
            qr.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="Scan to stream", use_container_width=True)
            st.code(phone_url, language="text")
            if st.button("Remove", key=f"rm_{vcam['session_id']}"):
                st.session_state.virtual_cams = [c for c in st.session_state.virtual_cams if c["session_id"] != vcam["session_id"]]
                st.session_state.tracking_active = False # Force worker reload
                st.rerun()

    # Android Chrome HTTPS workaround note
    if st.session_state.virtual_cams:
        with st.expander("⚠️ Android Chrome Camera Fix", expanded=False):
            st.markdown(f"""
**Phone camera not loading?**  
Modern mobile browsers block camera access over local HTTP connections. If you're on an Android phone, here is the official workaround:

1. Open Chrome on your phone and go to:  
   `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. Under "Insecure origins treated as secure", enter exactly:  
   **`{backend_url}`**
3. Change the dropdown from **Disabled** to **Enabled**.
4. Tap **Relaunch**.

Your phone will now allow the dashboard to access the camera over the local Wi-Fi!
            """)

    
    # Combine selected sources
    camera_sources = []
    for name in selected_local_names:
        camera_sources.append({"mode": "local", "source": local_cam_map[name], "name": name})
    for vcam in st.session_state.virtual_cams:
        camera_sources.append({"mode": "virtual", "source": vcam["session_id"], "name": vcam["name"]})
    
    st.markdown("---")
    view_options = ["Grid View (All Cameras)"] + [f"Focus: {src['name']}" for src in camera_sources]
    view_mode = st.selectbox("📺 View Mode", view_options)

    st.markdown("---")
    confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.50, 0.05)
    frame_interval_ms = st.slider("⏱️ Local Cam Interval (ms)", 0, 500, 30, 10, help="Only affects local webcams")

    st.markdown("---")
    box_thickness = st.slider("Box Thickness", 1, 5, 2)
    font_scale = st.slider("Label Font Scale", 0.3, 1.5, 0.6, 0.1)
    show_confidence = st.checkbox("Show Confidence %", value=True)

    st.markdown("---")
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            st.session_state.tracking_active = True
    with col_stop:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.tracking_active = False


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🎯 Multi-Camera Weapon Tracker</h1>
    <p>YOLOv8 + ByteTrack — Live Detection & Tracking Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop (Start/Stop Workers)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.tracking_active:
    
    # If workers are not started, initialize them
    if not st.session_state.workers:
        for src in camera_sources:
            worker = CameraWorker(
                mode=src["mode"],
                source=src["source"],
                name=src["name"],
                backend_url=backend_url,
                conf_thresh=confidence_threshold,
                box_thick=box_thickness,
                font_scale=font_scale,
                show_conf=show_confidence,
                frame_interval_ms=frame_interval_ms
            )
            worker.start()
            st.session_state.workers.append(worker)
            time.sleep(0.1)

    ui_container = st.empty()

    while st.session_state.tracking_active:
        with ui_container.container():
            if view_mode == "Grid View (All Cameras)":
                total_fps = sum(w.metrics['fps'] for w in st.session_state.workers)
                total_dets = sum(w.metrics['detections'] for w in st.session_state.workers)
                avg_inf = sum(w.metrics['inference_ms'] for w in st.session_state.workers) / max(1, len(st.session_state.workers))
                
                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-card"><div class="label">Cameras</div><div class="value">{len(st.session_state.workers)}</div></div>
                    <div class="metric-card"><div class="label">Total Detections</div><div class="value">{total_dets}</div></div>
                    <div class="metric-card"><div class="label">Avg FPS</div><div class="value">{total_fps/max(1, len(st.session_state.workers)):.1f}</div></div>
                    <div class="metric-card"><div class="label">Avg Inference</div><div class="value">{avg_inf:.0f}ms</div></div>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(max(1, len(st.session_state.workers)))
                for i, worker in enumerate(st.session_state.workers):
                    with cols[i]:
                        st.markdown(f"**{worker.name}** - {worker.status}")
                        if worker.latest_frame is not None:
                            st.image(worker.latest_frame, channels="RGB", use_container_width=True)
                        if worker.error_msg:
                            st.error(worker.error_msg)
                            
            else:
                focused_name = view_mode.replace("Focus: ", "")
                worker = next((w for w in st.session_state.workers if w.name == focused_name), None)
                if worker:
                    st.markdown(f"""
                    <div class="metrics-row">
                        <div class="metric-card"><div class="label">Camera</div><div class="value">{worker.name}</div></div>
                        <div class="metric-card"><div class="label">Status</div><div class="value" style="font-size: 1.2rem;">{worker.status}</div></div>
                        <div class="metric-card"><div class="label">Detections</div><div class="value">{worker.metrics['detections']}</div></div>
                        <div class="metric-card"><div class="label">FPS</div><div class="value">{worker.metrics['fps']:.1f}</div></div>
                        <div class="metric-card"><div class="label">Inference</div><div class="value">{worker.metrics['inference_ms']:.0f}ms</div></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if worker.latest_frame is not None:
                        st.image(worker.latest_frame, channels="RGB", use_container_width=True)
                    if worker.error_msg:
                        st.error(worker.error_msg)

        time.sleep(0.05)

else:
    if st.session_state.workers:
        for worker in st.session_state.workers:
            worker.stop()
        st.session_state.workers = []
        
    st.info("👈 Add your cameras in the sidebar and click **▶️ Start**.")
