import { useNavigate } from "react-router-dom";
import { ShieldCheck, Camera } from "lucide-react";

function UploadPage() {
  const navigate = useNavigate();

  return (
    <div className="page">
      <div className="container">

        {/* Header */}

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: 18,
          }}
        >
          <div
            style={{
              width: 70,
              height: 70,
              borderRadius: 20,
              background: "rgba(55,214,122,.12)",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              border: "1px solid rgba(55,214,122,.25)",
            }}
          >
            <ShieldCheck
              size={42}
              color="#37d67a"
              strokeWidth={2.5}
            />
          </div>

          <h1 className="page-title">
            Weapon Detection System
          </h1>
        </div>

        <p className="page-subtitle">
          AI-powered real-time weapon detection using YOLOv11
        </p>

        {/* Live Camera Card */}

        <div className="upload-grid">

          <div className="upload-card">

            <div className="upload-icon">
              <Camera
                size={72}
                color="#37d67a"
              />
            </div>

            <h2 className="upload-title">
              Live Camera Detection
            </h2>

            <p className="upload-description">
              Start real-time weapon detection using your webcam.
              The AI continuously monitors the live feed and
              automatically highlights detected weapons with
              bounding boxes.
            </p>

            <button
              className="upload-button"
              onClick={() => navigate("/live")}
            >
              Start Live Detection
            </button>

            <div className="file-types">

              <span className="file-type">
                LIVE
              </span>

              <span className="file-type">
                WEBCAM
              </span>

              <span className="file-type">
                YOLOv11
              </span>

            </div>

          </div>

        </div>

        <p className="footer-text">
          Built with FastAPI • YOLOv11 • React
        </p>

      </div>
    </div>
  );
}

export default UploadPage;