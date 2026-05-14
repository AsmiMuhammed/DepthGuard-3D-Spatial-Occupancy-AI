"""
test_pipeline.py
-----------------
Smoke-test the full pipeline without a webcam or video file.
Generates a synthetic scene (gradient + noise) and runs one depth
estimation + occupancy detection + visualisation pass.

Run:
    python test_pipeline.py
"""

import cv2
import numpy as np
import time

# ---- Generate a synthetic test frame ------------------------------------
print("[test] Creating synthetic 480×640 test frame …")
h, w = 480, 640

# Simulate a scene: gradient background + two bright blobs (simulated objects)
frame = np.zeros((h, w, 3), dtype=np.uint8)

# Gradient background
for i in range(h):
    frame[i, :] = [int(30 + 40 * i / h)] * 3   # dark-ish grey gradient

# "Object 1" – large, close (will be near/warm in depth map)
cv2.ellipse(frame, (200, 250), (120, 80), 0, 0, 360, (180, 140, 80), -1)

# "Object 2" – smaller, farther
cv2.ellipse(frame, (480, 150), (60, 50), 0, 0, 360, (80, 120, 180), -1)

# Add noise
noise = np.random.randint(0, 25, frame.shape, dtype=np.uint8)
frame = cv2.add(frame, noise)

cv2.imwrite("assets/test_input.png", frame)
print("[test] Saved  assets/test_input.png")

# ---- Depth estimation ---------------------------------------------------
print("[test] Running MiDaS depth estimation (this may take ~30s first run for download) …")
t0 = time.time()

from utils.depth import estimate_depth
depth_map = estimate_depth(frame, model_type="MiDaS_small")

elapsed = time.time() - t0
print(f"[test] Depth estimated in {elapsed*1000:.0f} ms")
print(f"[test] Depth map shape: {depth_map.shape}  min={depth_map.min():.3f}  max={depth_map.max():.3f}")

# ---- Occupancy detection ------------------------------------------------
from utils.occupancy import detect_occupancy
result = detect_occupancy(depth_map, warn_threshold=0.60, critical_threshold=0.80)

print(f"[test] Alert level   : {result.alert_level}")
print(f"[test] Occupied ratio: {result.occupied_ratio*100:.1f}%")
print(f"[test] Bounding boxes: {result.bounding_boxes}")

# ---- Visualisation ------------------------------------------------------
from utils.visualization import depth_to_colormap, draw_occupancy_overlay

depth_colour = depth_to_colormap(depth_map)
annotated    = draw_occupancy_overlay(frame, result)

cv2.imwrite("assets/test_depth.png",    depth_colour)
cv2.imwrite("assets/test_annotated.png", annotated)

print("[test] Saved  assets/test_depth.png")
print("[test] Saved  assets/test_annotated.png")
print("[test] ✅  Pipeline smoke-test PASSED")
