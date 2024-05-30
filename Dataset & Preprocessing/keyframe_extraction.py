import cv2
import numpy as np
import os

def extract_keyframes_farneback(video_path, output_folder, threshold=1.0, scale=0.5, frame_interval=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    keyframe_count = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        
        if frame_count % frame_interval != 0:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(magnitude)
        
        if mean_magnitude > threshold:
            keyframe_path = os.path.join(output_folder, f'motion_keyframe_farneback_{keyframe_count}.jpg')
            cv2.imwrite(keyframe_path, frame)
            keyframe_count += 1
            prev_gray = gray
    
    cap.release()

# Example usage
video_path = 'medium.mp4'
output_folder_farneback = 'keyframes_farneback'
extract_keyframes_farneback(video_path, output_folder_farneback, threshold=4.0, scale=0.5, frame_interval=5)
