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
import numpy as np
import easyocr
import time
import threading
import queue

def put_boxes(result,imageData):
  color = (0,0,255)
  imgHeight, imgWidth, _ = imageData.shape
  thick = 2
  font_scale = 1
  for res in result:
      top_left, btm_right = res[0][0],res[0][2]
      to_int = lambda items: [int(x) for x in items]
      top_left = to_int(top_left)
      btm_right = to_int(btm_right)
      
      label = res[1]
      
      print(label)

      cv2.rectangle(imageData,top_left, btm_right, color, thick)
      #cv2.putText(imageData, label, (top_left[0], top_left[1] - 12), 0, font_scale, color, thick)
  if len(result)>0:
    print('\n')


# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(0)#'v4l2src device=/dev/video0 ! video/x-raw, width=(int)640, height=(int)480, framerate=(fraction)30/1 ! videoconvert !  video/x-raw, format=(string)BGR ! appsink',cv2.CAP_GSTREAMER)
   
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video  file")
   

reader = easyocr.Reader(['en'], use_trt=True)
#if ur loading pth file for this, use 480x640! tis the frame.shape


      
iters = 0
time_labels = ['t_capture','t_infer', 't_box','t_show'] #[0.0009343624114990234, 1.134493112564087, 0.0008838176727294922, 0.0002529621124267578]
avg_times = [0]*len(time_labels)
times = []
result = []
q = queue.Queue()


# Read until video is completed
while(cap.isOpened()):
  times.append(time.time())
  # Capture frame-by-frame
  ret, frame = cap.read()
  times.append(time.time())
  #print('frame size', frame.shape)
#  print("num threads",threading.active_count())
  if ret == True:
    if True:#iters%10==0:
      if threading.active_count()==1:
        #result = reader.readtext(frame)
        threading.Thread(target=lambda *f: q.put(reader.readtext(np.array(f),text_threshold=.85)),args=(frame)).start()
      if not q.empty():
        result = q.get()
      times.append(time.time())
      #print(result)
      put_boxes(result,frame)
      times.append(time.time())
    # Display the resulting frame
      cv2.imshow('Frame', frame)
      times.append(time.time())
   
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
   
  # Break the loop
  else: 
    break
  iters+=1

  #perform calculations
  calc_avg = lambda t1,t2,t_old:((t2-t1))#+t_old) #/(2 if avg_times[0]!=0 else 1)
  avg_times = list(map(calc_avg,times[:-1],times[1:],avg_times))
#  print(avg_times,"fps:",1.0/(times[-1]-times[1]))
  times = []

   
# When everything done, release 
# the video capture object
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()


"""
[0.0012748241424560547, 1.2281830310821533, 0.0008254051208496094, 0.00030159950256347656] fps: 0.8134644401776536
Detection time: 0.2852518558502197
Total detection time 0.29549670219421387
Total Recogntion time 1.0111820697784424
Members Mark
out
Kancmao
Disinfecting 
WIPES
Toallitas Desiniecanle
Ws Cold & Flu Virus "
Kills 91.93,d brtntl
ORANGE SCEHT
AroMa A Maant
IuO BLCH
Oounav
(moh
foga tuEA La
Comat
dL 0s Nisos
quac4
78 WIPES
ZoWKbo 7eWc
rinimumw 1b
Dn
917
TaFe



IDEA:
have the read_text in a separate thread, and the rest in this thread. then video will be super smooth, but text will only show up choppy.
"""

