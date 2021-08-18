# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

#for video_capture.py
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python 
RUN pip3 install traitlets
RUN pip3 install scipy
RUN pip3 install tifffile

#for easyocr
RUN pip3 install python-bidi
ENV PYTHONIOENCODING=utf-8

#for torch2trt
RUN git clone --recursive -b jax-jp4.6.1-trt7 https://github.com/akamboj2/torch2trt.git torch2trt && \
    cd torch2trt && \
    python3 setup.py install && \
    cd ../ && \
    rm -rf torch2trt
