"""
This script generates synthetic anomaly videos from the UCF101 dataset.
It applies random temporal and spatial anomalies such as frame reversal, frame shuffling,
object insertions, and also combines clips from two different activities. It saves the generated
anomaly videos and records the anomalies and combined activities in a CSV file.
"""

import cv2
import os
import numpy as np
import csv
from glob import glob
import sys

input_video_dir = 'UCF-101/'
output_video_dir = 'output_videos/'
os.makedirs(output_video_dir, exist_ok=True)

os.makedirs('synthetic_objects', exist_ok=True)
for i in range(3):
    obj = np.zeros((50, 50, 4), dtype=np.uint8)
    color = np.random.randint(0, 255, size=3)
    obj[:, :, :3] = color
    obj[:, :, 3] = 255
    cv2.imwrite(f'synthetic_objects/object_{i}.png', obj)

object_paths = glob('synthetic_objects/*.png')

csv_file = open('anomaly_summary.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['video_name', 'original_activity', 'anomalies'])

n_samples = 100

videos_collected = []
activity_dirs = glob(os.path.join(input_video_dir, '*'))
for activity_dir in activity_dirs:
    activity_name = os.path.basename(activity_dir)
    video_files = glob(os.path.join(activity_dir, '*.mp4')) + glob(os.path.join(activity_dir, '*.avi'))
    for video_path in video_files:
        videos_collected.append((video_path, activity_name))

if len(videos_collected) < 2:
    print("Not enough videos found to combine.")
    csv_file.close()
    sys.exit(1)

np.random.shuffle(videos_collected)

for sample_idx in range(n_samples):
    primary_vid, primary_activity = videos_collected[sample_idx % len(videos_collected)]
    secondary_vid, secondary_activity = videos_collected[(sample_idx + 1) % len(videos_collected)]

    cap1 = cv2.VideoCapture(primary_vid)
    cap2 = cv2.VideoCapture(secondary_vid)
    frames = []

    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        frames.append(frame)
    cap1.release()

    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frames.append(frame)
    cap2.release()

    if not frames:
        continue

    h, w, _ = frames[0].shape
    out_name = f"anomaly_video_{sample_idx}.mp4"
    out_path = os.path.join(output_video_dir, out_name)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    anomalies = ['combined_activity_' + secondary_activity]

    if np.random.rand() < 0.3:
        frames = frames[::-1]
        anomalies.append('reversed')

    i = 0
    while i < len(frames):
        chunk = frames[i:i+10]

        if np.random.rand() < 0.3:
            np.random.shuffle(chunk)
            anomalies.append('shuffled')

        for j, f in enumerate(chunk):
            if np.random.rand() < 0.3:
                obj_path = np.random.choice(object_paths)
                obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
                x = np.random.randint(0, w-50)
                y = np.random.randint(0, h-50)
                alpha = obj[:, :, 3] / 255.0
                for c in range(3):
                    f[y:y+50, x:x+50, c] = (1-alpha) * f[y:y+50, x:x+50, c] + alpha * obj[:, :, c]
                anomalies.append('inserted_object')
            out.write(f)

        i += 10

    out.release()

    csv_writer.writerow([out_name, primary_activity, ';'.join(set(anomalies))])

csv_file.close()
