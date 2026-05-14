# generate_test_video.py
import cv2
import numpy as np

out = cv2.VideoWriter('sample_videos/test.mp4', 
                       cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 480))

for i in range(150):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate object approaching (growing circle)
    x = int(320 + 200 * np.sin(i * 0.1))
    r = int(20 + i * 0.6)
    cv2.circle(frame, (x, 240), r, (0, 200, 255), -1)
    
    # Second object
    cv2.rectangle(frame, (50, 50), (150 + i, 150), (100, 255, 100), -1)
    
    cv2.putText(frame, f'Frame {i}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    out.write(frame)

out.release()
print("Done! Upload sample_videos/test.mp4 in the app")