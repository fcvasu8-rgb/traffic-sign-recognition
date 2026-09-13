import React, { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  // =====================================================
  // IMAGE UPLOAD STATES
  // =====================================================

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // =====================================================
  // HISTORY
  // =====================================================

  const [history, setHistory] = useState([]);

  // =====================================================
  // CAMERA STATES
  // =====================================================

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraResult, setCameraResult] = useState(null);
  const [cameraLoading, setCameraLoading] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // =====================================================
  // IMAGE UPLOAD
  // =====================================================

  const handleImageChange = (file) => {
    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleFileInput = (event) => {
    const file = event.target.files[0];

    handleImageChange(file);
  };

  // =====================================================
  // DRAG AND DROP
  // =====================================================

  const handleDrop = (event) => {
    event.preventDefault();

    const file = event.dataTransfer.files[0];

    handleImageChange(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // =====================================================
  // IMAGE ANALYSIS
  // =====================================================

  const analyzeImage = async () => {
    if (!image) {
      alert("Please select a traffic sign image first.");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();

    formData.append("image", image);

    try {
      const response = await fetch(
        "http://localhost:5000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Prediction failed"
        );
      }

      setResult(data);

      const newPrediction = {
        prediction: data.prediction,
        confidence: data.confidence,
        time: new Date().toLocaleTimeString(),
      };

      setHistory((previousHistory) => [
        newPrediction,
        ...previousHistory,
      ]);

    } catch (error) {
      console.error(error);

      setResult({
        error:
          "Unable to connect to the prediction server. Make sure Flask is running.",
      });
    }

    setLoading(false);
  };

  // =====================================================
  // START CAMERA
  // =====================================================

  const startCamera = async () => {
    try {
      setCameraResult(null);

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",
          },
          audio: false,
        });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        await videoRef.current.play();
      }

      setCameraActive(true);

    } catch (error) {
      console.error(error);

      alert(
        "Unable to access camera. Please allow camera permission in your browser."
      );
    }
  };

  // =====================================================
  // STOP CAMERA
  // =====================================================

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  };

  // =====================================================
  // CAPTURE CAMERA IMAGE
  // =====================================================

  const captureAndAnalyze = async () => {
    if (!videoRef.current) {
      return;
    }

    if (!cameraActive) {
      alert("Please start the camera first.");
      return;
    }

    setCameraLoading(true);
    setCameraResult(null);

    const video = videoRef.current;

    const canvas = document.createElement("canvas");

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

    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          setCameraLoading(false);

          alert("Unable to capture camera image.");

          return;
        }

        const formData = new FormData();

        formData.append(
          "image",
          blob,
          "camera-frame.jpg"
        );

        try {
          const response = await fetch(
            "http://localhost:5000/predict-camera",
            {
              method: "POST",
              body: formData,
            }
          );

          const data = await response.json();

          if (!response.ok) {
            throw new Error(
              data.error ||
                "Camera prediction failed"
            );
          }

          setCameraResult(data);

          const newPrediction = {
            prediction: data.prediction,
            confidence: data.confidence,
            time: new Date().toLocaleTimeString(),
          };

          setHistory((previousHistory) => [
            newPrediction,
            ...previousHistory,
          ]);

        } catch (error) {
          console.error(error);

          setCameraResult({
            error:
              "Unable to connect to the camera prediction server.",
          });
        }

        setCameraLoading(false);
      },
      "image/jpeg",
      0.9
    );
  };

  // =====================================================
  // CLEAR HISTORY
  // =====================================================

  const clearHistory = () => {
    setHistory([]);
  };

  // =====================================================
  // STOP CAMERA WHEN PAGE CLOSES
  // =====================================================

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => {
            track.stop();
          });
      }
    };
  }, []);

  // =====================================================
  // UI
  // =====================================================

  return (
    <div className="app">

      {/* =================================================
          NAVBAR
      ================================================= */}

      <nav className="navbar">

        <div className="logo">
          🚦 TrafficAI
        </div>

        <div className="nav-links">

          <a href="#home">
            Home
          </a>

          <a href="#analyzer">
            Analyzer
          </a>

          <a href="#camera">
            Camera
          </a>

          <a href="#history">
            History
          </a>

          <a href="#about">
            About
          </a>

        </div>

      </nav>

      {/* =================================================
          HERO
      ================================================= */}

      <section
        id="home"
        className="hero"
      >

        <div className="hero-content">

          <span className="badge">
            AI POWERED TRAFFIC SIGN RECOGNITION
          </span>

          <h1>
            Understand Traffic Signs

            <span>
              Instantly with AI
            </span>
          </h1>

          <p>
            Upload a traffic sign image or use
            your camera and let our deep learning
            model identify the sign.
          </p>

          <a
            href="#analyzer"
            className="hero-button"
          >
            Start Recognition →
          </a>

        </div>

      </section>

      {/* =================================================
          IMAGE ANALYZER
      ================================================= */}

      <section
        id="analyzer"
        className="analyzer-section"
      >

        <div className="section-heading">

          <span>
            AI ANALYZER
          </span>

          <h2>
            Upload Traffic Sign
          </h2>

          <p>
            Select an image and our CNN model
            will analyze it.
          </p>

        </div>

        <div className="analyzer-card">

          <div
            className="upload-area"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >

            {preview ? (

              <img
                src={preview}
                alt="Traffic sign preview"
                className="preview-image"
              />

            ) : (

              <>
                <div className="upload-icon">
                  📷
                </div>

                <h3>
                  Drag & Drop Image
                </h3>

                <p>
                  or select an image from your computer
                </p>
              </>

            )}

            <label className="choose-button">

              Choose Image

              <input
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                hidden
              />

            </label>

          </div>

          <button
            className="analyze-button"
            onClick={analyzeImage}
            disabled={loading}
          >

            {loading ? (

              <>
                <span className="spinner"></span>
                Analyzing...
              </>

            ) : (

              <>
                🔍 Analyze Traffic Sign
              </>

            )}

          </button>

          {/* IMAGE RESULT */}

          {result && !result.error && (

            <div className="result-card">

              <div className="result-icon">
                🚦
              </div>

              <div className="result-content">

                <span className="result-label">
                  RECOGNIZED TRAFFIC SIGN
                </span>

                <h2>
                  {result.prediction}
                </h2>

                <div className="confidence-section">

                  <div className="confidence-header">

                    <span>
                      Confidence
                    </span>

                    <strong>
                      {result.confidence}%
                    </strong>

                  </div>

                  <div className="confidence-bar">

                    <div
                      className="confidence-fill"
                      style={{
                        width:
                          `${result.confidence}%`,
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

          {result && result.error && (

            <div className="error-message">

              ⚠️ {result.error}

            </div>

          )}

        </div>

      </section>

      {/* =================================================
          LIVE CAMERA
      ================================================= */}

      <section
        id="camera"
        className="camera-section"
      >

        <div className="section-heading">

          <span>
            LIVE AI CAMERA
          </span>

          <h2>
            Detect Traffic Signs Using Camera
          </h2>

          <p>
            Start your camera, show a traffic sign,
            capture it and let the CNN model recognize it.
          </p>

        </div>

        <div className="camera-card">

          {/* CAMERA VIEW */}

          <div className="camera-container">

            {cameraActive ? (

              <video
                ref={videoRef}
                className="camera-video"
                autoPlay
                playsInline
                muted
              ></video>

            ) : (

              <div className="camera-placeholder">

                <div className="camera-big-icon">
                  📷
                </div>

                <h3>
                  Camera is Off
                </h3>

                <p>
                  Click "Start Camera" to begin
                  traffic sign detection.
                </p>

              </div>

            )}

          </div>

          {/* CAMERA BUTTONS */}

          <div className="camera-buttons">

            {!cameraActive ? (

              <button
                className="camera-start-button"
                onClick={startCamera}
              >
                📷 Start Camera
              </button>

            ) : (

              <button
                className="camera-stop-button"
                onClick={stopCamera}
              >
                ⏹ Stop Camera
              </button>

            )}

            <button
              className="camera-capture-button"
              onClick={captureAndAnalyze}
              disabled={
                !cameraActive ||
                cameraLoading
              }
            >

              {cameraLoading ? (

                <>
                  <span className="spinner"></span>
                  Analyzing Camera...
                </>

              ) : (

                <>
                  📸 Capture & Analyze
                </>

              )}

            </button>

          </div>

          {/* CAMERA RESULT */}

          {cameraResult &&
            !cameraResult.error && (

              <div className="camera-result">

                <div className="camera-result-icon">
                  🚦
                </div>

                <div className="camera-result-content">

                  <span>
                    CAMERA DETECTION RESULT
                  </span>

                  <h2>
                    {cameraResult.prediction}
                  </h2>

                  <div className="confidence-header">

                    <span>
                      Confidence
                    </span>

                    <strong>
                      {cameraResult.confidence}%
                    </strong>

                  </div>

                  <div className="confidence-bar">

                    <div
                      className="confidence-fill"
                      style={{
                        width:
                          `${cameraResult.confidence}%`,
                      }}
                    ></div>

                  </div>

                  <p className="class-id">
                    Class ID:{" "}
                    {cameraResult.class_id}
                  </p>

                </div>

              </div>

            )}

          {cameraResult &&
            cameraResult.error && (

              <div className="error-message">

                ⚠️ {cameraResult.error}

              </div>

            )}

        </div>

      </section>

      {/* =================================================
          HISTORY
      ================================================= */}

      <section
        id="history"
        className="history-section"
      >

        <div className="section-heading">

          <span>
            PREDICTION HISTORY
          </span>

          <h2>
            Recent Predictions
          </h2>

          <p>
            View the traffic signs analyzed
            during this session.
          </p>

        </div>

        {history.length === 0 ? (

          <div className="empty-history">

            <div>
              📊
            </div>

            <h3>
              No predictions yet
            </h3>

            <p>
              Upload an image or use the camera
              to see your results here.
            </p>

          </div>

        ) : (

          <div className="history-container">

            <div className="history-top">

              <span>

                {history.length} prediction
                {history.length !== 1
                  ? "s"
                  : ""}

              </span>

              <button
                onClick={clearHistory}
                className="clear-button"
              >
                Clear History
              </button>

            </div>

            <div className="history-list">

              {history.map(
                (item, index) => (

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
                        {item.time}
                      </p>

                    </div>

                    <div className="history-confidence">

                      {item.confidence}%

                    </div>

                  </div>

                )
              )}

            </div>

          </div>

        )}

      </section>

      {/* =================================================
          HOW IT WORKS
      ================================================= */}

      <section className="how-section">

        <div className="section-heading">

          <span>
            WORKFLOW
          </span>

          <h2>
            How TrafficAI Works
          </h2>

        </div>

        <div className="steps">

          <div className="step">

            <div>
              📤
            </div>

            <h3>
              1. Upload
            </h3>

            <p>
              Upload a traffic sign image.
            </p>

          </div>

          <div className="step">

            <div>
              📷
            </div>

            <h3>
              2. Camera
            </h3>

            <p>
              Capture a traffic sign using
              your camera.
            </p>

          </div>

          <div className="step">

            <div>
              🧠
            </div>

            <h3>
              3. CNN Analysis
            </h3>

            <p>
              The trained deep learning model
              analyzes the image.
            </p>

          </div>

          <div className="step">

            <div>
              🎯
            </div>

            <h3>
              4. Prediction
            </h3>

            <p>
              The model identifies the traffic
              sign and confidence score.
            </p>

          </div>

        </div>

      </section>

      {/* =================================================
          ABOUT
      ================================================= */}

      <section
        id="about"
        className="about-section"
      >

        <div className="about-card">

          <span>
            COLLEGE PROJECT
          </span>

          <h2>
            Traffic Sign Recognition
            Using Deep Learning
          </h2>

          <p>
            This project uses a Convolutional
            Neural Network trained on the GTSRB
            traffic sign dataset to automatically
            recognize traffic signs from images
            and camera captures.
          </p>

          <div className="tech-list">

            <span>
              React
            </span>

            <span>
              Python
            </span>

            <span>
              Flask
            </span>

            <span>
              TensorFlow
            </span>

            <span>
              Keras
            </span>

            <span>
              CNN
            </span>

          </div>

        </div>

      </section>

      {/* =================================================
          FOOTER
      ================================================= */}

      <footer>

        <p>
          © 2026 TrafficAI | Traffic Sign
          Recognition Using Deep Learning
        </p>

      </footer>

    </div>
  );
}

export default App;