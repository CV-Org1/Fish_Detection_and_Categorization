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
out = cv.VideoWriter('d_output_video.avi', fourcc, fps, (frame_width, frame_height))

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

    # Calculate dense optical flow using Farneback method
    flow_farneback = cv.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Sparse optical flow - Lucas-Kanade with pyramidal implementation
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_pts = cv.goodFeaturesToTrack(prev_frame_gray, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=30)
    next_pts, status, error = cv.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, prev_pts, None, **lk_params)
    good_old = prev_pts[status == 1]
    good_new = next_pts[status == 1]
    img_lk_pyramid = frame.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        img_lk_pyramid = cv.circle(img_lk_pyramid, (int(a), int(b)), 5, (0, 255, 0), -1)

    # Sparse optical flow - Sparse RLOF (Robust Local Optical Flow)
    rof = cv.optflow.createOptFlow_DeepFlow()
    flow_rlof = rof.calc(prev_frame_gray, frame_gray, None)
    hsv_rlof = np.zeros_like(frame)
    hsv_rlof[..., 1] = 255
    mag_rlof, ang_rlof = cv.cartToPolar(flow_rlof[..., 0], flow_rlof[..., 1])
    hsv_rlof[..., 0] = ang_rlof * 180 / np.pi / 2
    hsv_rlof[..., 2] = cv.normalize(mag_rlof, None, 0, 255, cv.NORM_MINMAX)
    flow_rgb_rlof = cv.cvtColor(hsv_rlof, cv.COLOR_HSV2BGR)
    img_rlof = cv.addWeighted(frame, 0.5, flow_rgb_rlof, 1, 0)

    # Write the frame to the video file
    out.write(img_rlof)

    frames_to_display.append(img_rlof)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    prev_frame_gray = frame_gray

cap.release()
out.release()
cv.destroyAllWindows()

# Display frames as small grids
display_frames_grid(frames_to_display)
