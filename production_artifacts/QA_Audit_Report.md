# 🔍 QA Audit Report — Real-Time Weapon Tracking App

> **Auditor**: QA Engineer (@qa)  
> **Date**: 2026-07-07  
> **Verdict**: ✅ **PASS — Green Light to Run**

---

## Audit Scope

| File | Path | Lines |
|------|------|-------|
| Backend Server | `app_build/backend/main.py` | 287 |
| Backend Dependencies | `app_build/backend/requirements.txt` | 18 |
| Frontend App | `app_build/frontend/app.py` | 516 |
| Frontend Dependencies | `app_build/frontend/requirements.txt` | 15 |

---

## Bugs Found & Fixed

### 🔴 BUG-001: Dual WebSocket Decorator Silently Drops `/ws` Route (CRITICAL)

**File**: `backend/main.py` (lines 215-216, original)

**Problem**: Stacking two `@app.websocket()` decorators on the same function is **not supported** by FastAPI. Only the last decorator (`/ws/track`) was registered — the original `/ws` endpoint was silently lost.

```python
# ❌ BROKEN — only /ws/track was actually registered
@app.websocket("/ws")
@app.websocket("/ws/track")
async def websocket_endpoint(websocket: WebSocket):
```

**Fix**: Split into two separate endpoint functions that delegate to a shared `_handle_tracking_session()` coroutine:

```python
# ✅ FIXED — both routes now work independently
async def _handle_tracking_session(websocket: WebSocket):
    ...

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await _handle_tracking_session(websocket)

@app.websocket("/ws/track")
async def websocket_track_endpoint(websocket: WebSocket):
    await _handle_tracking_session(websocket)
```

---

### 🟡 BUG-002: Deprecated `@app.on_event("startup")` (MEDIUM)

**File**: `backend/main.py` (line 78, original)

**Problem**: `@app.on_event("startup")` is deprecated since FastAPI 0.104+ and emits deprecation warnings. The modern replacement is the `lifespan` async context manager.

**Fix**: Replaced with `@asynccontextmanager async def lifespan(app)` and passed it to `FastAPI(lifespan=lifespan)`.

---

### 🟡 BUG-003: Blocking YOLO Inference on Async Event Loop (MEDIUM)

**File**: `backend/main.py` (line 257, original)

**Problem**: `run_inference()` is a synchronous, CPU/GPU-bound function (~30-200ms per call) that was called directly inside an `async` WebSocket handler. This **blocks the entire asyncio event loop**, preventing the server from accepting new connections or processing other I/O during inference.

**Fix**: Wrapped the call in `asyncio.to_thread()`:

```python
# ✅ Non-blocking — inference runs in a thread pool
response = await asyncio.to_thread(run_inference_sync, frame, frame_count)
```

---

### 🟡 BUG-004: Incorrect Exception Class Reference (MEDIUM)

**File**: `frontend/app.py` (line 496, original)

**Problem**: `websocket.WebSocketConnectionClosedException` is not a top-level attribute of the `websocket` module in the `websocket-client` package. The correct import path is `websocket._exceptions.WebSocketConnectionClosedException`. Using the wrong path would cause an `AttributeError` at runtime when the server closes the connection, leading to an unhandled crash instead of a clean warning.

**Fix**: Added explicit import:

```python
from websocket._exceptions import WebSocketConnectionClosedException
```

---

### 🟡 BUG-005: No Timeout on `ws.recv()` — Potential Infinite Hang (MEDIUM)

**File**: `frontend/app.py` (line 445, original)

**Problem**: `ws.recv()` was called with no timeout. If the backend stalls, hangs, or takes excessively long, the Streamlit process blocks indefinitely with no way to recover.

**Fix**: Added `ws.settimeout(10)` before each recv call (10-second timeout).

---

### 🟢 BUG-006: Missing Streamlit Port in CORS Origins (LOW)

**File**: `backend/main.py` (CORS config, original)

**Problem**: CORS origins only included Vite (`5173`) and React (`3000`) ports. Streamlit runs on port `8501` by default — any fetch/XHR from the Streamlit frontend to the health endpoint would be blocked.

**Fix**: Added `http://localhost:8501` and `http://127.0.0.1:8501` to `allow_origins`.

---

## Verification Checklist

| Category | Check | Result |
|----------|-------|--------|
| **Dependencies** | All `import` statements have matching entries in `requirements.txt` | ✅ Pass |
| **Dependencies** | `fastapi`, `uvicorn[standard]`, `ultralytics`, `opencv-python`, `numpy`, `websockets` in backend | ✅ Pass |
| **Dependencies** | `streamlit`, `opencv-python`, `numpy`, `websocket-client` in frontend | ✅ Pass |
| **Dependencies** | No version conflicts between backend and frontend (shared `opencv-python`, `numpy`) | ✅ Pass |
| **WebSocket** | `/ws` route registered and functional | ✅ Fixed (BUG-001) |
| **WebSocket** | `/ws/track` route registered and functional | ✅ Pass |
| **WebSocket** | Graceful `WebSocketDisconnect` handling in backend | ✅ Pass |
| **WebSocket** | Connection timeout on frontend `create_connection()` | ✅ Fixed (10s) |
| **WebSocket** | Recv timeout on frontend `ws.recv()` | ✅ Fixed (BUG-005) |
| **WebSocket** | `WebSocketConnectionClosedException` caught correctly | ✅ Fixed (BUG-004) |
| **JSON Protocol** | Client sends `{"frame": "<base64>"}` | ✅ Pass |
| **JSON Protocol** | Server returns `{"detections": [...], "inference_time_ms": float, "frame_count": int}` | ✅ Pass |
| **JSON Protocol** | Each detection has `bbox`, `track_id`, `confidence`, `class_name` | ✅ Pass |
| **JSON Protocol** | Frontend correctly parses all 4 detection fields with `.get()` defaults | ✅ Pass |
| **ByteTrack** | `model.track(persist=True, tracker='bytetrack.yaml')` used | ✅ Pass |
| **ByteTrack** | `boxes.id` null-checked before accessing tracking IDs | ✅ Pass |
| **Rendering** | `cv2.rectangle()` and `cv2.putText()` used for bounding boxes + labels | ✅ Pass |
| **Rendering** | BGR → RGB conversion before `st.image()` | ✅ Pass |
| **Error Handling** | Base64 decode errors → JSON error response (no crash) | ✅ Pass |
| **Error Handling** | Invalid JSON from client → error response (no crash) | ✅ Pass |
| **Error Handling** | Missing `frame` field → error response (no crash) | ✅ Pass |
| **Error Handling** | Webcam open failure → clean error + stop | ✅ Pass |
| **Error Handling** | WS connection failure → clean error + stop | ✅ Pass |
| **Async Safety** | YOLO inference runs in thread pool (non-blocking) | ✅ Fixed (BUG-003) |
| **Resource Cleanup** | `cap.release()` and `ws.close()` in `finally` block | ✅ Pass |

---

## Verdict

> **✅ ALL CLEAR — You are green-lit to run both servers.**

All 6 bugs have been fixed in place. The codebase is clean, dependency-complete, and production-ready for local execution.
