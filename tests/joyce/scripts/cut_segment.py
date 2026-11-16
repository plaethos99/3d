"""Copy a frame range of a video to a new file, re-encoded, with the same frame rate.

Usage: cut_segment.py <video> <out.mp4> <first_frame> <last_frame>
"""
import sys

import cv2

video, out = sys.argv[1:3]
first, last = int(sys.argv[3]), int(sys.argv[4])
capture = cv2.VideoCapture(video)
fps = capture.get(cv2.CAP_PROP_FPS)
writer = None
index = 0
written = 0
while index <= last:
    ok, frame = capture.read()
    if not ok:
        break
    if index >= first:
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        writer.write(frame)
        written += 1
    index += 1
capture.release()
writer.release()
print("CUT_OK frames", written, "fps", round(fps, 2))
