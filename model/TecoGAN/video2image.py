import argparse
import os

import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()
video = cv.VideoCapture(args.input)
fps = video.get(5)
print(fps)
count = 1
while True:
    _, frame = video.read()
    if frame is None:
        break
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    savepath = os.path.join(args.output, str(count) + '.png')
    cv.imwrite(savepath, frame)
    count += 1
video.release()