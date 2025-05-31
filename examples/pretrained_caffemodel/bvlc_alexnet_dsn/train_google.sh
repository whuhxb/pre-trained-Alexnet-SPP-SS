#!/usr/bin/env sh
export PYTHONPATH=$PWD/python:$PWD/examples/pretrained_caffemodel/bvlc_alexnet_dsn

./build/tools/caffe train -solver /home/stu_3/Documents/caffe/examples/pretrained_caffemodel/bvlc_alexnet_dsn/bvlc_alexnet_dsn_solver_google.prototxt  -weights /home/stu_3/Documents/caffe/examples/pretrained_caffemodel/bvlc_alexnet_dsn_v1/bvlc_alexnet.caffemodel