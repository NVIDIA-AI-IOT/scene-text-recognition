"""
The MIT License (MIT)

Copyright (c) 2021 NVIDIA CORPORATION

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import cv2
import easyocr
import time
import threading

from jetvision.jetvision.camera import Camera


def put_boxes(result, imageData):
    color = (0, 0, 255)
    imgHeight, imgWidth, _ = imageData.shape
    thick = 2
    font_scale = 1
    for res in result:
        top_left, btm_right = res[0][0], res[0][2]
        to_int = lambda items: [int(x) for x in items]
        top_left = to_int(top_left)
        btm_right = to_int(btm_right)

        label = res[1]

        print(label)

        cv2.rectangle(imageData, top_left, btm_right, color, thick)
        # cv2.putText(imageData, label, (top_left[0], top_left[1] - 12), 0, font_scale, color, thick)
    if len(result) > 0:
        print("\n")


def main():

    print("Setting up camera...")
    cap = Camera(0, shape_in=(1920, 1080), shape_out=(224, 224))

    print("Loading model...")
    reader = easyocr.Reader(["en"], use_trt=False)

    # Read until video is completed
    for _ in range(int(1e5)):
        frame = cap.read()
        # frame = cv2.resize(frame, (224,224))
        result = reader.readtext(frame, text_threshold=0.85)
        print(result)
        # put_boxes(result, frame)

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.stop()


if __name__ == "__main__":
    main()
