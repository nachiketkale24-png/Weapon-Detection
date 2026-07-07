"""
Real-Time Weapon Tracking App — Streamlit Frontend
====================================================
A Streamlit application that:
  1. Captures live webcam frames locally via OpenCV.
  2. Sends base64-encoded JPEG frames to the FastAPI backend over WebSocket.
  3. Receives JSON detections (bounding boxes, tracking IDs, confidences, class names).
  4. Draws bounding boxes and labels on the frame using OpenCV.
  5. Displays the annotated frame in the Streamlit UI in real time.

Usage:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import base64
import json
import time
import websocket
from websocket._exceptions import WebSocketConnectionClosedException

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🔫 Weapon Tracker — Real-Time Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for a polished dark-themed UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ── */
    .stApp {
        background-color: #0e1117;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2rem;
        margin: 0 0 0.3rem 0;
        font-weight: 700;
    }
    .main-header p {
        color: #8892b0;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Metrics Row ── */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(233, 69, 96, 0.15);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        flex: 1;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card .label {
        color: #8892b0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #e94560;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-connected {
        background: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    .status-disconnected {
        background: rgba(255, 68, 68, 0.1);
        color: #ff4444;
        border: 1px solid rgba(255, 68, 68, 0.3);
    }
    .status-idle {
        background: rgba(255, 193, 7, 0.1);
        color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] h2 {
        color: #e94560;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────

if "tracking_active" not in st.session_state:
    st.session_state.tracking_active = False
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "frames_processed" not in st.session_state:
    st.session_state.frames_processed = 0

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Settings Panel
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    # WebSocket URL configuration
    ws_url = st.text_input(
        "🌐 WebSocket URL",
        value="ws://localhost:8000/ws/track",
        help="The WebSocket endpoint of the FastAPI backend server.",
    )

    st.markdown("---")

    # Confidence threshold slider
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Detections below this confidence score will be hidden.",
    )

    # Frame send interval
    frame_interval_ms = st.slider(
        "⏱️ Frame Interval (ms)",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        help="Delay between sending consecutive frames (lower = faster, higher CPU).",
    )

    st.markdown("---")

    # Bounding box visual settings
    st.markdown("### 🎨 Display Options")
    box_thickness = st.slider("Box Thickness", 1, 5, 2)
    font_scale = st.slider("Label Font Scale", 0.3, 1.5, 0.6, 0.1)
    show_confidence = st.checkbox("Show Confidence %", value=True)

    st.markdown("---")

    # Start / Stop controls
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶️ Start", use_container_width=True, type="primary")
    with col_stop:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)

    if start_btn:
        st.session_state.tracking_active = True
    if stop_btn:
        st.session_state.tracking_active = False

# ─────────────────────────────────────────────────────────────────────────────
# Color Palette for Bounding Boxes (indexed by track_id)
# ─────────────────────────────────────────────────────────────────────────────

BOX_COLORS = [
    (0, 0, 255),     # Red
    (0, 165, 255),   # Orange
    (0, 255, 255),   # Yellow
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Cyan
    (128, 0, 255),   # Rose
    (0, 128, 255),   # Deep Orange
    (255, 128, 0),   # Sky Blue
]


def get_box_color(track_id: int | None) -> tuple:
    """Return a consistent color for a given track_id."""
    if track_id is None:
        return (0, 0, 255)  # Default red for untracked
    return BOX_COLORS[track_id % len(BOX_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing Utility
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    conf_threshold: float,
    thickness: int,
    f_scale: float,
    draw_confidence: bool,
) -> tuple[np.ndarray, int]:
    """
    Draw bounding boxes and labels on the frame for each detection.

    Args:
        frame:            BGR image (numpy array).
        detections:       List of detection dicts from the backend.
        conf_threshold:   Minimum confidence to render.
        thickness:        Rectangle border thickness.
        f_scale:          Font scale for labels.
        draw_confidence:  Whether to append confidence % to the label.

    Returns:
        Annotated frame and count of drawn detections.
    """
    drawn_count = 0

    for det in detections:
        confidence = det.get("confidence", 0.0)

        # Filter by confidence threshold
        if confidence < conf_threshold:
            continue

        # Parse bounding box coordinates
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(c) for c in bbox]
        track_id = det.get("track_id")
        class_name = det.get("class_name", "unknown")

        # Assign a consistent color per track_id
        color = get_box_color(track_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        id_str = f"ID:{track_id}" if track_id is not None else "ID:?"
        label = f"{id_str} {class_name}"
        if draw_confidence:
            label += f" {confidence:.0%}"

        # Compute label background size for readability
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 1
        )
        label_y = max(y1 - 10, text_h + 5)

        # Draw label background rectangle
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 5),
            (x1 + text_w + 6, label_y + baseline),
            color,
            cv2.FILLED,
        )

        # Draw label text (white on colored background)
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        drawn_count += 1

    return frame, drawn_count


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🎯 Real-Time Weapon Tracker</h1>
    <p>YOLOv8 + ByteTrack — Live Detection &amp; Tracking Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics Row Placeholder
# ─────────────────────────────────────────────────────────────────────────────

metrics_placeholder = st.empty()
status_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Video Display Placeholder
# ─────────────────────────────────────────────────────────────────────────────

frame_placeholder = st.empty()
info_placeholder = st.empty()


def render_metrics(fps: float, det_count: int, frame_num: int, inf_time: float):
    """Render the live metrics row using custom HTML."""
    metrics_placeholder.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="label">FPS</div>
            <div class="value">{fps:.1f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Detections</div>
            <div class="value">{det_count}</div>
        </div>
        <div class="metric-card">
            <div class="label">Frames</div>
            <div class="value">{frame_num}</div>
        </div>
        <div class="metric-card">
            <div class="label">Inference</div>
            <div class="value">{inf_time:.0f}ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status(status: str):
    """Render a connection status badge."""
    css_class = {
        "connected": "status-connected",
        "disconnected": "status-disconnected",
        "idle": "status-idle",
    }.get(status, "status-idle")

    dot = {"connected": "🟢", "disconnected": "🔴", "idle": "🟡"}.get(status, "🟡")
    label = status.upper()

    status_placeholder.markdown(
        f'<div class="status-badge {css_class}">{dot} {label}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Tracking Loop
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.tracking_active:
    # ── Idle State ──
    render_status("idle")
    render_metrics(fps=0.0, det_count=0, frame_num=0, inf_time=0.0)
    info_placeholder.info(
        "👈 Configure settings in the sidebar and click **▶️ Start** to begin tracking."
    )

else:
    # ── Active Tracking ──
    info_placeholder.empty()

    # --- Open Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        render_status("disconnected")
        st.error("❌ Could not access the webcam. Please check your camera permissions.")
        st.session_state.tracking_active = False
        st.stop()

    # --- Connect to WebSocket ---
    ws = None
    try:
        ws = websocket.create_connection(ws_url, timeout=10)
        render_status("connected")
    except Exception as e:
        render_status("disconnected")
        st.error(f"❌ Could not connect to WebSocket at `{ws_url}`.\n\n**Error:** {e}")
        cap.release()
        st.session_state.tracking_active = False
        st.stop()

    # --- Tracking Loop ---
    frame_count = 0
    fps = 0.0
    prev_time = time.perf_counter()

    try:
        while st.session_state.tracking_active:
            loop_start = time.perf_counter()

            # 1. Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to capture frame from webcam.")
                break

            # 2. Encode frame as base64 JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            success, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                continue

            b64_frame = base64.b64encode(buffer).decode("utf-8")

            # 3. Send frame over WebSocket
            message = json.dumps({"frame": b64_frame})
            ws.send(message)

            # 4. Receive detection response
            ws.settimeout(10)  # 10s recv timeout to avoid hanging
            response_raw = ws.recv()
            response = json.loads(response_raw)

            # Check for server-side errors
            if "error" in response:
                st.warning(f"⚠️ Server error: {response['error']}")
                continue

            detections = response.get("detections", [])
            inference_time = response.get("inference_time_ms", 0.0)
            server_frame_count = response.get("frame_count", 0)

            # 5. Draw bounding boxes and labels on the frame
            annotated_frame, drawn_count = draw_detections(
                frame=frame,
                detections=detections,
                conf_threshold=confidence_threshold,
                thickness=box_thickness,
                f_scale=font_scale,
                draw_confidence=show_confidence,
            )

            # 6. Convert BGR → RGB for Streamlit display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # 7. Display the annotated frame
            frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

            # 8. Calculate FPS
            frame_count += 1
            current_time = time.perf_counter()
            elapsed = current_time - prev_time
            if elapsed >= 0.5:  # Update FPS every 500ms for stability
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = current_time

            # 9. Update metrics display
            render_metrics(
                fps=fps,
                det_count=drawn_count,
                frame_num=server_frame_count,
                inf_time=inference_time,
            )

            # 10. Respect the frame interval to control send rate
            elapsed_ms = (time.perf_counter() - loop_start) * 1000
            sleep_ms = max(0, frame_interval_ms - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    except WebSocketConnectionClosedException:
        render_status("disconnected")
        st.warning("⚠️ WebSocket connection was closed by the server.")

    except Exception as e:
        render_status("disconnected")
        st.error(f"❌ An error occurred during tracking: {e}")

    finally:
        # Cleanup resources
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        cap.release()
        st.session_state.tracking_active = False
        render_status("disconnected")
