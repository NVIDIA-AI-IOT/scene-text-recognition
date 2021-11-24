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

from camera import Camera

def put_boxes(result, arr, text=False):
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

        # Draw BB
        cv2.rectangle(arr, top_left, btm_right, color, thick)
        
        # Draw text
        if text:
            cv2.putText(arr, label, (top_left[0], top_left[1] - 12), 0, font_scale, color, thick)

def main():

    DISPLAY = False

    print("Setting up camera...")
    cam = Camera(0, shape_in=(1920, 1080), shape_out=(224, 224))

    print("Loading model...")
    reader = easyocr.Reader(["en"], use_trt=True)

    for _ in range(1000):
        
        arr = cam.read()
        result = reader.readtext(arr, text_threshold=0.85)
        print(result)

        # Display the resulting frame
        if DISPLAY:
            cv2.imshow("Frame", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

if __name__ == "__main__":
    main()
