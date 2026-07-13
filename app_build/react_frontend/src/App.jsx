import { Routes, Route } from "react-router-dom";

import UploadPage from "./pages/UploadPage";
import LiveCameraPage from "./pages/LiveCameraPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />

      <Route path="/live" element={<LiveCameraPage />} />
    </Routes>
  );
}

export default App;