# 🛡️ DepthGuard — 3D Spatial Occupancy AI

> **An AI-powered real-time depth estimation and spatial occupancy monitoring system for industrial safety.**

---

## 📌 Project Overview

DepthGuard is a computer-vision system that uses **MiDaS deep-learning depth estimation** to analyse video/webcam feeds and detect when objects or people enter unsafe proximity zones. It is designed to simulate an industrial-grade spatial safety monitoring tool.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 MiDaS Depth Estimation | Intel MiDaS (small / DPT-Hybrid / DPT-Large) |
| 📏 Occupancy Detection | Threshold-based unsafe zone identification |
| 🎨 Depth Heatmap | INFERNO / JET / TURBO colourmap visualisation |
| ⚠️ Three-tier Alerts | SAFE → WARNING → CRITICAL |
| 📦 Bounding Boxes | Connected-component blob detection |
| 📸 Multiple Input Modes | Image upload, Video upload, Webcam live stream |
| 📊 Dashboard Metrics | Occupied %, closest distance, FPS, frame count |
| 🗂️ Alert Log | Tabular history of all triggered alerts |
| 🔧 Adjustable Thresholds | Live sidebar sliders for all parameters |

---

## 🏗️ Architecture

```
Camera / Video Input
        │
        ▼
Frame Extraction (OpenCV)
        │
        ▼
MiDaS Depth Estimation (PyTorch)
        │
        ▼
Depth Map (float32, normalised 0–1)
        │
        ▼
Occupancy Detection Logic
   ├── warn_mask   (depth ≥ warn_threshold)
   └── critical_mask (depth ≥ critical_threshold)
        │
        ▼
Blob Detection (OpenCV connectedComponents)
        │
        ▼
Visualisation (overlays, bounding boxes, HUD)
        │
        ▼
Streamlit Dashboard
```

---

## 📁 Folder Structure

```
DepthGuard/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt
├── README.md
│
├── utils/
│   ├── __init__.py
│   ├── depth.py            ← MiDaS depth estimation wrapper
│   ├── occupancy.py        ← Occupancy detection + alert logic
│   └── visualization.py   ← OpenCV drawing helpers
│
├── assets/                 ← Static assets (logos, etc.)
├── models/                 ← Optional: cached model weights
├── sample_videos/          ← Place test videos here
└── logs/
    └── depthguard.log      ← Runtime alert log (auto-created)
```

---

## ⚙️ Installation

### 1. Clone / copy the project

```bash
git clone https://github.com/your-username/DepthGuard.git
cd DepthGuard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU support**: If you have an NVIDIA GPU, install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🎮 Usage

1. Select an **Input Source** in the sidebar (Image / Video / Webcam)
2. Choose a **MiDaS model variant** (MiDaS_small for speed)
3. Adjust **Warning** and **Critical** thresholds with the sliders
4. Upload your file or start the webcam
5. Watch real-time depth maps, occupancy overlays, and alerts

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Depth Estimation | MiDaS (intel-isl/MiDaS via torch.hub) |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| UI / Dashboard | Streamlit |
| Numerical | NumPy |
| Logging | Python logging |

---

## 🏭 Industrial Use Cases

- **Smart Factories**: Detect workers entering robot operating zones
- **Warehouse Automation**: Prevent forklifts approaching pedestrians
- **Construction Sites**: Monitor unsafe proximity to heavy machinery
- **Retail**: Crowd density and flow management
- **Robotics**: Collaborative robot (cobot) safety envelope monitoring

---

## 🔮 Future Improvements

- [ ] YOLOv8 person detection integration for person-specific alerts
- [ ] Edge AI deployment (ONNX export, TensorRT, OpenVINO)
- [ ] MQTT/WebSocket alert streaming to PLC / SCADA systems
- [ ] Multi-camera support and bird's-eye view stitching
- [ ] Historical analytics and shift-level safety reports
- [ ] 3D point cloud reconstruction from stereo depth

---

## 📸 Screenshots

> _Add screenshots of the running dashboard here._

---

## 📄 Licence

MIT — free to use for hackathons, research, and prototyping.
