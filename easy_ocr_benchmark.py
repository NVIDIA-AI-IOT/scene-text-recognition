import easyocr
import torch
import torch.nn as nn
from torch2trt import torch2trt
import time
import os

#torch2trt custom converters
from torch2trt import *
@tensorrt_converter('torch.Tensor.__hash__')
@tensorrt_converter('torch.Tensor.get_device')
@tensorrt_converter('torch.Tensor.data_ptr')
@tensorrt_converter('torch.Tensor.is_complex')
@tensorrt_converter('torch.is_grad_enabled')
def suppress_warning(ctx):
    #none of these effect the computational path thus don't need converters
    pass

@tensorrt_converter('torch.zeros')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    output = ctx.method_return
    output._trt = add_missing_trt_tensors(ctx.network, [output])

def profile(model,dummy_input):
    iters=10
    with torch.no_grad():
            # warm up
            for _ in range(10):
                model(dummy_input)

            # throughput evaluate
            torch.cuda.current_stream().synchronize()
            t0 = time.time()
            for _ in range(iters):
                model(dummy_input)
            torch.cuda.current_stream().synchronize()
            t1 = time.time()
            throughput = 1.0 * iters / (t1 - t0)

            # latency evaluate
            torch.cuda.current_stream().synchronize()
            t0 = time.time()
            for _ in range(iters):
                model(dummy_input)
                torch.cuda.current_stream().synchronize()
            t1 = time.time()
            latency = round(1000.0 * (t1 - t0) / iters, 2)
    print("throughput: %.3f \t latency: %.3f"% (throughput,latency))

if __name__ == '__main__':

    reader = easyocr.Reader(['en'],gpu=True) # need to run only once to load model into memory

    if not os.path.exists('torch2trt_models'):
        os.makedirs('torch2trt_models')

    #detector: 
    y = torch.ones((1, 3, 224, 224),dtype=torch.float).cuda()
    print("Detector:")
    print("Before Conversion:")
    profile(reader.detector, y) #throughput: 12.386 	 latency: 84.190

    if os.path.isfile('torch2trt_models/easyocr_detect.pth'):
        model_trt_detect = TRTModule()
        model_trt_detect.load_state_dict(torch.load('torch2trt_models/easyocr_detect.pth'))
    else:
        model_trt_detect = torch2trt(reader.detector, [y])
        torch.save(model_trt_detect.state_dict(),'torch2trt_models/easyocr_detect.pth')

    print("After Conversion")
    profile(model_trt_detect, y) #throughput: 24.737 	 latency: 48.990


    #recognizer
    print("\nRecognizer:")
    x = torch.ones((1,1,64,320),dtype=torch.float).to('cuda')
    reader.recognizer.eval()
    print("Before Conversion:")
    profile(reader.recognizer, x) #throughput: 36.912 	 latency: 24.610

    if os.path.isfile('torch2trt_models/easyocr_recognize.pth'):
        model_trt_rec = TRTModule()
        model_trt_rec.load_state_dict(torch.load('torch2trt_models/easyocr_recognize.pth'))
    else:
        model_trt_rec = torch2trt(reader.detector, [y])
        torch.save(model_trt_rec.state_dict(),'torch2trt_models/easyocr_recognize.pth')
    model_trt_rec = torch2trt(reader.recognizer, [x])#, use_onnx=True)

    print("After Conversion")
    profile(model_trt_rec,x) #throughput: 2296.110 	 latency: 0.450
    torch.save(model_trt_rec.state_dict(),'torch2trt_models/easyocr_recognize.pth')
