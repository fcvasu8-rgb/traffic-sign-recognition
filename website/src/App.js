import React, { useEffect, useRef, useState } from "react";
import "./App.css";

// ===============================
// DEPLOYED FLASK BACKEND
// ===============================
const API_URL =
  "https://traffic-sign-recognition-d120.onrender.com";

function App() {
  // ===============================
  // STATES
  // ===============================
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [history, setHistory] = useState([]);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraResult, setCameraResult] = useState(null);
  const [cameraLoading, setCameraLoading] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // ===============================
  // IMAGE FILE SELECTION
  // ===============================
  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setResult(null);

    const imageURL = URL.createObjectURL(file);
    setPreview(imageURL);
  };

  // ===============================
  // DRAG AND DROP
  // ===============================
  const handleDrop = (event) => {
    event.preventDefault();

    const file = event.dataTransfer.files[0];

    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file.");
      return;
    }

    setSelectedFile(file);
    setResult(null);

    const imageURL = URL.createObjectURL(file);
    setPreview(imageURL);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // ===============================
  // UPLOAD IMAGE TO FLASK
  // ===============================
  const predictImage = async () => {
    if (!selectedFile) {
      alert("Please select an image first.");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();

      formData.append("image", selectedFile);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();

      setResult(data);

      // Add prediction to history
      const newHistory = {
        type: "Image Upload",
        prediction: data.prediction,
        confidence: data.confidence,
        class_id: data.class_id,
        time: new Date().toLocaleString(),
      };

      setHistory((prev) => [newHistory, ...prev]);
    } catch (error) {
      console.error(error);

      alert(
        "Unable to connect to the prediction server.\n\n" +
        "Please make sure the Render backend is running."
      );
    }

    setLoading(false);
  };

  // ===============================
  // CLEAR IMAGE
  // ===============================
  const clearImage = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
  };

  // ===============================
  // START CAMERA
  // ===============================
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraActive(true);
      setCameraResult(null);
    } catch (error) {
      console.error(error);

      alert(
        "Camera access denied or unavailable.\n\n" +
        "Please allow camera permission in your browser."
      );
    }
  };

  // ===============================
  // STOP CAMERA
  // ===============================
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  };

  // ===============================
  // CAPTURE CAMERA IMAGE
  // ===============================
  const captureImage = async () => {
    if (!videoRef.current || !canvasRef.current) {
      return;
    }

    setCameraLoading(true);

    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.toBlob(async (blob) => {
      if (!blob) {
        setCameraLoading(false);
        return;
      }

      try {
        const formData = new FormData();

        formData.append(
          "image",
          blob,
          "camera-capture.jpg"
        );

        // ===============================
        // DEPLOYED FLASK CAMERA API
        // ===============================
        const response = await fetch(
          `${API_URL}/predict-camera`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (!response.ok) {
          throw new Error("Camera prediction failed");
        }

        const data = await response.json();

        setCameraResult(data);

        // Add camera result to history
        const newHistory = {
          type: "Camera",
          prediction: data.prediction,
          confidence: data.confidence,
          class_id: data.class_id,
          time: new Date().toLocaleString(),
        };

        setHistory((prev) => [newHistory, ...prev]);
      } catch (error) {
        console.error(error);

        alert(
          "Unable to connect to the camera prediction server."
        );
      }

      setCameraLoading(false);
    }, "image/jpeg");
  };

  // ===============================
  // STOP CAMERA WHEN PAGE CLOSES
  // ===============================
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  // ===============================
  // SCROLL TO SECTION
  // ===============================
  const scrollToSection = (id) => {
    const section = document.getElementById(id);

    if (section) {
      section.scrollIntoView({
        behavior: "smooth",
      });
    }
  };

  // ===============================
  // UI
  // ===============================
  return (
    <div className="app">

      {/* =====================================
          NAVBAR
      ===================================== */}
      <nav className="navbar">
        <div className="logo">
          🚦 Traffic Sign AI
        </div>

        <div className="nav-links">
          <button onClick={() => scrollToSection("home")}>
            Home
          </button>

          <button onClick={() => scrollToSection("analyzer")}>
            Analyzer
          </button>

          <button onClick={() => scrollToSection("camera")}>
            Camera
          </button>

          <button onClick={() => scrollToSection("history")}>
            History
          </button>

          <button onClick={() => scrollToSection("about")}>
            About
          </button>
        </div>
      </nav>

      {/* =====================================
          HERO SECTION
      ===================================== */}
      <section id="home" className="hero">

        <div className="hero-content">

          <div className="hero-badge">
            🧠 Powered by Deep Learning
          </div>

          <h1>
            Traffic Sign
            <span> Recognition</span>
          </h1>

          <p>
            Identify traffic signs instantly using a
            Convolutional Neural Network trained on the
            GTSRB traffic sign dataset.
          </p>

          <div className="hero-buttons">

            <button
              className="primary-btn"
              onClick={() => scrollToSection("analyzer")}
            >
              Analyze Image
            </button>

            <button
              className="secondary-btn"
              onClick={() => scrollToSection("camera")}
            >
              Open Camera
            </button>

          </div>

        </div>

      </section>

      {/* =====================================
          ANALYZER SECTION
      ===================================== */}
      <section id="analyzer" className="section">

        <div className="section-title">

          <span>IMAGE ANALYZER</span>

          <h2>
            Upload a Traffic Sign
          </h2>

          <p>
            Upload an image and our CNN model will
            identify the traffic sign.
          </p>

        </div>

        <div className="analyzer-card">

          {/* UPLOAD AREA */}
          <div
            className="upload-area"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >

            {!preview ? (

              <div className="upload-content">

                <div className="upload-icon">
                  📷
                </div>

                <h3>
                  Drag & Drop Image
                </h3>

                <p>
                  or select an image from your device
                </p>

                <label className="upload-btn">
                  Choose Image

                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    hidden
                  />
                </label>

              </div>

            ) : (

              <div className="preview-container">

                <img
                  src={preview}
                  alt="Traffic sign preview"
                  className="preview-image"
                />

                <div className="preview-actions">

                  <button
                    className="primary-btn"
                    onClick={predictImage}
                    disabled={loading}
                  >
                    {loading
                      ? "Analyzing..."
                      : "🔍 Analyze Sign"}
                  </button>

                  <button
                    className="clear-btn"
                    onClick={clearImage}
                  >
                    Clear
                  </button>

                </div>

              </div>

            )}

          </div>

          {/* RESULT */}
          {result && (

            <div className="result-card">

              <div className="result-icon">
                🚦
              </div>

              <div className="result-content">

                <p className="result-label">
                  RECOGNIZED TRAFFIC SIGN
                </p>

                <h2>
                  {result.prediction}
                </h2>

                <div className="confidence">

                  <div className="confidence-header">
                    <span>
                      Confidence
                    </span>

                    <strong>
                      {Number(result.confidence).toFixed(2)}%
                    </strong>
                  </div>

                  <div className="confidence-bar">

                    <div
                      className="confidence-fill"
                      style={{
                        width: `${Math.min(
                          Number(result.confidence),
                          100
                        )}%`,
                      }}
                    ></div>

                  </div>

                </div>

                <p className="class-id">
                  Class ID: {result.class_id}
                </p>

              </div>

            </div>

          )}

        </div>

      </section>

      {/* =====================================
          CAMERA SECTION
      ===================================== */}
      <section id="camera" className="section camera-section">

        <div className="section-title">

          <span>LIVE CAMERA</span>

          <h2>
            Detect Using Camera
          </h2>

          <p>
            Point your camera toward a traffic sign
            and capture it for prediction.
          </p>

        </div>

        <div className="camera-card">

          <div className="camera-container">

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-video"
            />

            {!cameraActive && (

              <div className="camera-placeholder">

                <div className="camera-icon">
                  📷
                </div>

                <h3>
                  Camera is Off
                </h3>

                <p>
                  Start the camera to detect traffic signs.
                </p>

              </div>

            )}

          </div>

          <canvas
            ref={canvasRef}
            style={{ display: "none" }}
          />

          <div className="camera-buttons">

            {!cameraActive ? (

              <button
                className="primary-btn"
                onClick={startCamera}
              >
                📷 Start Camera
              </button>

            ) : (

              <>
                <button
                  className="primary-btn"
                  onClick={captureImage}
                  disabled={cameraLoading}
                >
                  {cameraLoading
                    ? "Analyzing..."
                    : "📸 Capture & Analyze"}
                </button>

                <button
                  className="clear-btn"
                  onClick={stopCamera}
                >
                  Stop Camera
                </button>
              </>

            )}

          </div>

          {/* CAMERA RESULT */}
          {cameraResult && (

            <div className="result-card camera-result">

              <div className="result-icon">
                🚦
              </div>

              <div className="result-content">

                <p className="result-label">
                  CAMERA PREDICTION
                </p>

                <h2>
                  {cameraResult.prediction}
                </h2>

                <div className="confidence">

                  <div className="confidence-header">

                    <span>
                      Confidence
                    </span>

                    <strong>
                      {Number(
                        cameraResult.confidence
                      ).toFixed(2)}%
                    </strong>

                  </div>

                  <div className="confidence-bar">

                    <div
                      className="confidence-fill"
                      style={{
                        width: `${Math.min(
                          Number(cameraResult.confidence),
                          100
                        )}%`,
                      }}
                    ></div>

                  </div>

                </div>

                <p className="class-id">
                  Class ID: {cameraResult.class_id}
                </p>

              </div>

            </div>

          )}

        </div>

      </section>

      {/* =====================================
          WORKFLOW SECTION
      ===================================== */}
      <section className="workflow-section">

        <div className="section-title">

          <span>HOW IT WORKS</span>

          <h2>
            From Image to Prediction
          </h2>

        </div>

        <div className="workflow">

          <div className="workflow-step">

            <div className="step-number">
              01
            </div>

            <div className="step-icon">
              📷
            </div>

            <h3>
              Upload
            </h3>

            <p>
              Upload a traffic sign image.
            </p>

          </div>

          <div className="workflow-line"></div>

          <div className="workflow-step">

            <div className="step-number">
              02
            </div>

            <div className="step-icon">
              🧠
            </div>

            <h3>
              CNN Processing
            </h3>

            <p>
              The deep learning model analyzes the image.
            </p>

          </div>

          <div className="workflow-line"></div>

          <div className="workflow-step">

            <div className="step-number">
              03
            </div>

            <div className="step-icon">
              🎯
            </div>

            <h3>
              Prediction
            </h3>

            <p>
              The system identifies the traffic sign.
            </p>

          </div>

        </div>

      </section>

      {/* =====================================
          HISTORY SECTION
      ===================================== */}
      <section id="history" className="section">

        <div className="section-title">

          <span>PREDICTION HISTORY</span>

          <h2>
            Recent Predictions
          </h2>

        </div>

        <div className="history-card">

          {history.length === 0 ? (

            <div className="empty-history">

              <div>
                📋
              </div>

              <h3>
                No predictions yet
              </h3>

              <p>
                Your predictions will appear here.
              </p>

            </div>

          ) : (

            <div className="history-list">

              {history.map((item, index) => (

                <div
                  className="history-item"
                  key={index}
                >

                  <div className="history-icon">
                    🚦
                  </div>

                  <div className="history-info">

                    <h3>
                      {item.prediction}
                    </h3>

                    <p>
                      {item.type} • {item.time}
                    </p>

                  </div>

                  <div className="history-confidence">

                    <strong>
                      {Number(
                        item.confidence
                      ).toFixed(2)}%
                    </strong>

                    <span>
                      confidence
                    </span>

                  </div>

                </div>

              ))}

            </div>

          )}

        </div>

      </section>

      {/* =====================================
          ABOUT SECTION
      ===================================== */}
      <section id="about" className="about-section">

        <div className="about-content">

          <div className="section-title">

            <span>ABOUT THE PROJECT</span>

            <h2>
              Traffic Sign Recognition
              Using Deep Learning
            </h2>

          </div>

          <p>
            This project uses a Convolutional Neural
            Network (CNN) to recognize traffic signs
            from images. The model is trained using
            the German Traffic Sign Recognition
            Benchmark (GTSRB) dataset.
          </p>

          <p>
            The system can recognize different
            categories including speed limits,
            stop signs, warning signs, pedestrian
            crossings and other road signs.
          </p>

          <div className="technology-grid">

            <div className="technology">
              <span>🐍</span>
              <strong>Python</strong>
              <small>Backend</small>
            </div>

            <div className="technology">
              <span>🔥</span>
              <strong>TensorFlow</strong>
              <small>Deep Learning</small>
            </div>

            <div className="technology">
              <span>⚛️</span>
              <strong>React</strong>
              <small>Frontend</small>
            </div>

            <div className="technology">
              <span>🚀</span>
              <strong>Flask</strong>
              <small>API</small>
            </div>

          </div>

        </div>

      </section>

      {/* =====================================
          FOOTER
      ===================================== */}
      <footer className="footer">

        <div>
          🚦 Traffic Sign AI
        </div>

        <p>
          Traffic Sign Recognition Using Deep Learning
        </p>

        <span>
          Built with React + Flask + TensorFlow
        </span>

      </footer>

    </div>
  );
}

export default App;