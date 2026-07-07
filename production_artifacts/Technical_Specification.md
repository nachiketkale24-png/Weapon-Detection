# 📋 Technical Specification: Real-Time Weapon Tracking App

> **Version**: 1.0  
> **Author**: Product Manager (@pm)  
> **Date**: 2026-07-07  
> **Status**: 🟡 Awaiting User Approval

---

## 1. Executive Summary

This application is a **real-time weapon detection and tracking dashboard** that leverages a teammate's pre-trained **YOLOv8 object detection model**. It provides a live webcam feed in the browser, streams video frames to a Python backend via WebSockets, runs inference with persistent object tracking (ByteTrack), and renders bounding boxes with tracking IDs directly on the user's screen in real time.

The system is designed for **low-latency, high-throughput** operation suitable for security monitoring, research demonstrations, and surveillance prototyping.

---

## 2. Requirements

### 2.1 Functional Requirements

| ID    | Requirement                                                                                                  | Priority |
|-------|--------------------------------------------------------------------------------------------------------------|----------|
| FR-01 | The frontend SHALL access the user's webcam via the browser's `MediaDevices` API.                            | P0       |
| FR-02 | The frontend SHALL capture video frames and encode them as **base64 JPEG** images.                           | P0       |
| FR-03 | The frontend SHALL send encoded frames to the backend via a **WebSocket** connection at a configurable interval (default: **100ms / ~10 FPS**). | P0       |
| FR-04 | The backend SHALL receive base64-encoded frames, decode them into OpenCV-compatible NumPy arrays.            | P0       |
| FR-05 | The backend SHALL run YOLOv8 inference using `.track(persist=True, tracker='bytetrack.yaml')` for persistent multi-object tracking. | P0       |
| FR-06 | The backend SHALL extract **bounding boxes** (x1, y1, x2, y2), **tracking IDs**, **confidence scores**, and **class names** from the model output. | P0       |
| FR-07 | The backend SHALL return detections as a **JSON message** over the same WebSocket connection.                | P0       |
| FR-08 | The frontend SHALL render bounding boxes and tracking IDs on an **HTML5 `<canvas>`** element absolutely positioned over the `<video>` element. | P0       |
| FR-09 | The frontend SHALL display a **connection status indicator** (connected / disconnected / reconnecting).      | P1       |
| FR-10 | The frontend SHALL display a **live FPS counter** showing the processing frame rate.                         | P1       |
| FR-11 | The frontend SHALL provide a **confidence threshold slider** to filter low-confidence detections.            | P2       |

### 2.2 Non-Functional Requirements

| ID     | Requirement                                                                                          | Priority |
|--------|------------------------------------------------------------------------------------------------------|----------|
| NFR-01 | End-to-end latency (frame capture → bounding box render) SHALL be **< 500ms** on a machine with a supported GPU. | P0       |
| NFR-02 | The backend SHALL handle **at least 1 concurrent WebSocket client**.                                 | P0       |
| NFR-03 | The application SHALL be runnable **entirely locally** with no cloud dependencies.                   | P0       |
| NFR-04 | The frontend SHALL be **responsive** and work on screens ≥ 1024px wide.                              | P1       |
| NFR-05 | The codebase SHALL follow clean, well-documented structure with separation of concerns.              | P1       |

---

## 3. Architecture & Tech Stack

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                           │
│                                                                 │
│  ┌──────────┐    base64 frame     ┌──────────────────────────┐  │
│  │  Webcam   │ ──────────────────► │  React Frontend          │  │
│  │  (Media   │                     │  - <video> element       │  │
│  │  Devices) │                     │  - <canvas> overlay      │  │
│  └──────────┘                      │  - WebSocket client      │  │
│                                    └──────────┬───────────────┘  │
│                                               │                  │
└───────────────────────────────────────────────┼──────────────────┘
                                                │ WebSocket (ws://)
                                                │ JSON ↑↓ base64
┌───────────────────────────────────────────────┼──────────────────┐
│                     BACKEND SERVER            │                  │
│                                               ▼                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  FastAPI + WebSocket Endpoint                               │ │
│  │  POST /ws                                                   │ │
│  │                                                             │ │
│  │  1. Receive base64 frame                                    │ │
│  │  2. Decode → OpenCV ndarray                                 │ │
│  │  3. YOLOv8 .track(persist=True, tracker='bytetrack.yaml')  │ │
│  │  4. Extract boxes, IDs, confidences, class names            │ │
│  │  5. Return JSON: { detections: [...] }                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Dependencies: ultralytics, opencv-python, fastapi, uvicorn      │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Tech Stack Decision Matrix

| Layer       | Technology              | Rationale                                                                  |
|-------------|-------------------------|----------------------------------------------------------------------------|
| **Backend** | Python 3.10+            | Native ecosystem for ML/CV workloads; required by `ultralytics`.           |
| **Backend** | FastAPI                 | Async-native, first-class WebSocket support, auto-generated docs.          |
| **Backend** | Uvicorn                 | ASGI server; performant and production-ready for FastAPI.                   |
| **Backend** | Ultralytics (YOLOv8)    | Teammate's pre-trained model; `.track()` with ByteTrack built-in.          |
| **Backend** | OpenCV (`cv2`)          | Industry standard for frame decoding/encoding; pairs with NumPy.           |
| **Frontend**| React 18+ (Vite)        | Component-based; excellent for canvas/video composition and state updates. |
| **Frontend**| Vanilla CSS             | Full control over styling; no framework overhead for a focused UI.         |
| **Protocol**| WebSockets              | Full-duplex, low-latency; ideal for real-time streaming use cases.         |

### 3.3 Project File Structure

```
app_build/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket endpoint, YOLO inference
│   ├── requirements.txt         # Python dependencies
│   └── bytetrack.yaml           # ByteTrack tracker configuration (if custom needed)
│
├── frontend/
│   ├── index.html               # Vite entry point
│   ├── package.json             # Node dependencies
│   ├── vite.config.js           # Vite configuration (proxy for WebSocket)
│   ├── src/
│   │   ├── main.jsx             # React entry point
│   │   ├── App.jsx              # Root component
│   │   ├── App.css              # Global styles
│   │   ├── components/
│   │   │   ├── VideoCanvas.jsx  # Webcam + Canvas overlay component
│   │   │   ├── VideoCanvas.css  # Styles for the video/canvas container
│   │   │   ├── StatusBar.jsx    # Connection status + FPS display
│   │   │   ├── StatusBar.css    # Styles for the status bar
│   │   │   ├── Controls.jsx     # Confidence slider & controls
│   │   │   └── Controls.css     # Styles for controls
│   │   └── hooks/
│   │       ├── useWebSocket.js  # Custom hook: WebSocket lifecycle management
│   │       └── useWebcam.js     # Custom hook: Webcam access & frame capture
│   └── public/
│       └── favicon.svg          # App favicon
```

---

## 4. API / WebSocket Protocol

### 4.1 WebSocket Endpoint

**URL**: `ws://localhost:8000/ws`

### 4.2 Client → Server Message (Frame)

```json
{
  "frame": "<base64-encoded-JPEG-string>"
}
```

### 4.3 Server → Client Message (Detections)

```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "track_id": 1,
      "confidence": 0.92,
      "class_name": "knife"
    },
    {
      "bbox": [x1, y1, x2, y2],
      "track_id": 3,
      "confidence": 0.87,
      "class_name": "gun"
    }
  ],
  "inference_time_ms": 45.2,
  "frame_count": 142
}
```

- **`bbox`**: Pixel coordinates `[x1, y1, x2, y2]` relative to the original frame dimensions.
- **`track_id`**: Persistent integer ID assigned by ByteTrack across frames.
- **`confidence`**: Float 0.0–1.0 from the YOLOv8 model.
- **`class_name`**: Human-readable class label from the model.
- **`inference_time_ms`**: Server-side inference duration for performance monitoring.
- **`frame_count`**: Monotonically increasing counter for frame synchronization.

---

## 5. State Management & Data Flow

### 5.1 Frontend Data Flow

```
┌──────────┐   getUserMedia()   ┌──────────┐   drawImage()    ┌──────────┐
│  Webcam  │ ────────────────►  │  <video>  │ ────────────────► │ <canvas> │
│  Stream  │                    │  element  │   (background)    │ overlay  │
└──────────┘                    └─────┬─────┘                   └────▲─────┘
                                      │                              │
                            setInterval(100ms)                       │
                                      │                              │
                                      ▼                              │
                              ┌──────────────┐    JSON response      │
                              │ Capture Frame │ ─────────────────────►│
                              │ → base64     │    detections[]       │
                              │ → send(ws)   │                       │
                              └──────────────┘    drawRect() +       │
                                                  fillText()         │
```

### 5.2 React State Model

| State Variable       | Type            | Managed By          | Purpose                                      |
|----------------------|-----------------|---------------------|----------------------------------------------|
| `detections`         | `Array<Object>` | `App.jsx`           | Current frame's detection results             |
| `isConnected`        | `boolean`       | `useWebSocket`      | WebSocket connection status                   |
| `fps`                | `number`        | `useWebSocket`      | Calculated frames per second                  |
| `confidenceThreshold`| `number`        | `Controls.jsx`      | Min confidence to render (default: 0.5)       |
| `webcamReady`        | `boolean`       | `useWebcam`         | Whether the webcam stream is active           |

### 5.3 Backend State

- **YOLOv8 Model**: Loaded once at startup and reused across all inference calls. The `persist=True` flag ensures ByteTrack maintains tracking state across sequential frames within a session.
- **Frame Counter**: A simple integer counter incremented per processed frame, returned in each response for client-side synchronization.

---

## 6. Key Implementation Notes

### 6.1 Backend
- The YOLOv8 model path should be configurable via an environment variable or a constant (defaulting to `best.pt` or `yolov8n.pt`).
- Frame decoding: `base64.b64decode()` → `np.frombuffer()` → `cv2.imdecode()`.
- CORS must be enabled for local development (`http://localhost:5173`).
- The WebSocket handler should be wrapped in a `try/except` for graceful disconnection handling.

### 6.2 Frontend
- The `<canvas>` must have `position: absolute` and be overlaid pixel-perfectly on the `<video>` element using matching `width` and `height`.
- Canvas coordinates must be scaled if the video's natural resolution differs from its display size.
- Bounding boxes should use distinct colors (e.g., red for high confidence, yellow for medium) and display the tracking ID label.
- The `useWebSocket` hook should implement auto-reconnection with exponential backoff.

---

> **⚠️ APPROVAL GATE**: This document requires your explicit approval before development begins.
