import { useNavigate } from "react-router-dom";
import {
  Camera,
  ShieldCheck,
  ArrowLeft,
  Activity,
  Cpu,
  Eye,
} from "lucide-react";

function LiveCameraPage() {
  const navigate = useNavigate();

  return (
    <div className="page">

      <div className="container">

        <div className="live-header">

          <div className="live-title">

            <ShieldCheck
              size={42}
              color="#37d67a"
            />

            <div>

              <h1 className="page-title">
                Live Weapon Detection
              </h1>

              <p className="page-subtitle">
                AI-powered real-time surveillance using YOLOv11
              </p>

            </div>

          </div>

          <button
            className="back-button"
            onClick={() => navigate("/")}
          >
            <ArrowLeft size={18}/>
            Back
          </button>

        </div>

        <div className="live-layout">

          {/* Camera Feed */}

          <div className="live-card">

            <div className="live-card-header">

              <Camera
                size={22}
                color="#37d67a"
              />

              <span>Live Camera Feed</span>

              <div className="live-badge">
                ● LIVE
              </div>

            </div>

            <img
              src="http://127.0.0.1:8000/video_feed"
              alt="Live Camera"
              className="live-stream"
            />

          </div>
{/* Security Status */}

<div className="analytics-panel">

  <div className="analytics-title">

    <ShieldCheck
      size={22}
      color="#37d67a"
    />

    Security Status

  </div>

  <div className="live-stat">

    <Camera
      size={18}
      color="#37d67a"
    />

    <div>

      <h4>Camera</h4>

      <p>Online</p>

    </div>

  </div>

  <div className="live-stat">

    <Cpu
      size={18}
      color="#37d67a"
    />

    <div>

      <h4>AI Model</h4>

      <p>YOLOv11 Loaded</p>

    </div>

  </div>

  <div className="live-stat">

    <Activity
      size={18}
      color="#37d67a"
    />

    <div>

      <h4>Detection</h4>

      <p>Active</p>

    </div>

  </div>

  <div className="live-stat">

    <Eye
      size={18}
      color="#37d67a"
    />

    <div>

      <h4>Monitoring</h4>

      <p>Live Camera Feed</p>

    </div>

  </div>

  <div className="live-note">

    YOLOv11 continuously processes the live camera feed.
    Weapon detections are displayed directly on the video stream.

  </div>

</div>

        </div>

      </div>

    </div>
  );
}

export default LiveCameraPage;