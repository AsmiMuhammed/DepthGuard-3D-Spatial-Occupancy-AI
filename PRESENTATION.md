# 🛡️ DepthGuard — Hackathon Presentation

---

## 🔴 Problem Statement

Modern industrial workplaces — factories, warehouses, construction sites — face a critical safety challenge: **workers and machines share physical space**, and traditional sensor-based proximity detection is expensive, inflexible, and requires fixed infrastructure.

> Every year, thousands of workplace accidents occur due to **unsafe proximity** between humans, robots, and heavy equipment — most of which could be prevented with real-time spatial awareness.

**Current solutions are:**
- Expensive LiDAR sensors (€5,000–€50,000 per unit)
- Fixed laser curtains with no semantic understanding
- Wired installations with high maintenance cost
- Zero AI intelligence — binary trip-wire only

---

## 💡 Solution Overview

**DepthGuard** is a camera-only, AI-powered spatial occupancy monitoring system that:

1. Takes a standard RGB camera feed (any webcam or CCTV)
2. Estimates a **full scene depth map** using Intel MiDaS (monocular depth estimation)
3. Identifies **occupied zones** based on configurable proximity thresholds
4. Issues **three-tier alerts** (SAFE / WARNING / CRITICAL) in real time
5. Runs on a **standard laptop CPU** — no expensive hardware required

**Key innovation**: Replace costly sensor arrays with a single camera + AI model.

---

## 🔄 Workflow

```
📷 Camera / Video Input
        │
        ▼
🖼️  Frame Extraction
        │
        ▼
🧠  MiDaS Monocular Depth Estimation
        │   (intel-isl/MiDaS, PyTorch)
        ▼
🗺️  Normalised Depth Map  (0 = far, 1 = near)
        │
        ▼
📐  Occupancy Detection Logic
        ├── depth ≥ warn_threshold  →  WARNING zone
        └── depth ≥ critical_threshold  →  CRITICAL zone
        │
        ▼
🔵  Blob Detection (connected components)
        │
        ▼
🎨  Visual Overlays + Bounding Boxes + HUD
        │
        ▼
📊  Streamlit Safety Dashboard
        ├── Live annotated video feed
        ├── Depth heatmap
        ├── Alert badge (SAFE / WARNING / CRITICAL)
        ├── Metrics panel (occupied %, FPS, closest distance)
        └── Alert history log
```

---

## 🏭 Industrial Applications

| Sector | Application |
|---|---|
| **Smart Factory** | Detect workers entering robot end-effector reach zones |
| **Warehouse** | Alert when pedestrians approach automated guided vehicles (AGVs) |
| **Construction** | Monitor exclusion zones around cranes and excavators |
| **Healthcare** | Patient fall-risk monitoring in wards |
| **Retail** | Queue and crowd density management |
| **Robotics** | Collaborative robot safety envelope enforcement |

---

## 🛠️ Technical Stack

| Layer | Technology | Why |
|---|---|---|
| Depth AI | MiDaS (DPT / MiDaS_small) | State-of-the-art monocular depth, runs on CPU |
| Framework | PyTorch | Production-grade, torch.hub model loading |
| Vision | OpenCV | Industry-standard CV library |
| Dashboard | Streamlit | Rapid professional UI, no frontend code needed |
| Numerics | NumPy | Fast array operations |
| Logging | Python logging | Persistent safety audit trail |

---

## 📊 Alert Tiers

| Level | Trigger Condition | Visual Indicator |
|---|---|---|
| ✅ **SAFE** | No depth breach | Green banner |
| ⚠️ **WARNING** | Object closer than warn_threshold | Orange overlay + bounding boxes |
| 🚨 **CRITICAL** | Object closer than critical_threshold OR occupied area > N% | Pulsing red overlay + banner |

All alerts are logged to `logs/depthguard.log` with timestamps.

---

## 🔮 Future Scope

### Short-term (Sprint 2)
- **YOLOv8 integration**: Distinguish *person* vs *object* for person-specific alerts
- **ONNX export**: Run on Raspberry Pi / Jetson Nano (Edge AI)
- **Alert sound**: Audio buzzer on CRITICAL events

### Medium-term
- **Stereo camera support**: True metric depth (centimetres) instead of relative
- **3D occupancy grid**: Voxel-based spatial map
- **MQTT integration**: Push alerts to factory PLC / SCADA systems

### Long-term
- **Digital twin**: Mirror the physical safety zones in a 3D simulation
- **Predictive safety**: Trajectory forecasting to warn before breach occurs
- **Multi-site cloud dashboard**: Centralised safety analytics across facilities

---

## 🎬 Demo Explanation

### What the demo shows:
1. Upload a sample video (or use webcam)
2. Depth map is generated frame-by-frame in real time
3. Objects close to the camera are highlighted in **orange/red**
4. Alert badge changes from ✅ SAFE → ⚠️ WARNING → 🚨 CRITICAL
5. Bounding boxes appear around occupied regions
6. Metrics panel shows: occupied %, closest depth %, FPS
7. Alert log captures every triggered event

### Key demo talking point:
> *"This entire system runs on a standard laptop CPU with a USB webcam — no LiDAR, no depth camera, no special hardware. Just a camera and AI."*

---

## 🏆 Competitive Advantages

- ✅ **Zero special hardware** — works with any camera
- ✅ **Real-time** — MiDaS_small runs at 5–15 FPS on CPU
- ✅ **Configurable** — all thresholds adjustable live from UI
- ✅ **Lightweight** — deployable on edge devices
- ✅ **Open source** — based on publicly available models
- ✅ **Extensible** — plug in YOLO, MQTT, cloud logging

---

*DepthGuard · Built for [AI Challenge Name] · 48-hour Hackathon*
