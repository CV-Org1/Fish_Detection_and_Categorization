import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load the video
video_path = "D:\dhaya\GX010096.mp4"
cap = cv.VideoCapture(video_path)

# Check if the video capture was successful
if not cap.isOpened():
    print('Error opening video file!')
    exit()

# Read the first frame to get frame dimensions
ret, first_frame = cap.read()
if not ret:
    print('Error reading the first frame!')
    exit()

frame_height, frame_width, _ = first_frame.shape

# Create a mask image for drawing purposes
mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Function to display frames as small grids
def display_frames_grid(frames, grid_size=(4, 4)):
    num_frames = len(frames)
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    axes = axes.flatten()

    for i in range(min(num_frames, len(axes))):
        axes[i].imshow(cv.cvtColor(frames[i], cv.COLOR_BGR2RGB))
        axes[i].axis('off')

    for ax in axes[num_frames:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Lists to store frames for display and video writing
frames_to_display = []
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

# Read the first frame again
cap = cv.VideoCapture(video_path)

ret, prev_frame = cap.read()
prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Dense optical flow - SimpleFlow
    sflow = cv.optflow.createOptFlow_SimpleFlow()
    flow_simple = sflow.calc(prev_frame_gray, frame_gray, None)
    hsv_simple = np.zeros_like(frame)
    hsv_simple[..., 1] = 255
    mag_simple, ang_simple = cv.cartToPolar(flow_simple[..., 0], flow_simple[..., 1])
    hsv_simple[..., 0] = ang_simple * 180 / np.pi / 2
    hsv_simple[..., 2] = cv.normalize(mag_simple, None, 0, 255, cv.NORM_MINMAX)
    flow_rgb_simple = cv.cvtColor(hsv_simple, cv.COLOR_HSV2BGR)
    img_simple = cv.addWeighted(frame, 0.5, flow_rgb_simple, 1, 0)

    # Write the frame to the video file
    out.write(img_simple)

    frames_to_display.append(img_simple)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    prev_frame_gray = frame_gray

cap.release()
out.release()
cv.destroyAllWindows()

# Display frames as small grids
display_frames_grid(frames_to_display)
