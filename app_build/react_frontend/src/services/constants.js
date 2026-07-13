// Backend URL
export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// Live Camera Stream Endpoint
export const VIDEO_FEED_URL = `${API_BASE_URL}/video_feed`;