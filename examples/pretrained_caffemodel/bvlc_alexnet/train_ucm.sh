#!/usr/bin/env sh
export PYTHONPATH=$PWD/python:$PWD/examples/pretrained_caffemodel/bvlc_alexnet
./build/tools/caffe train -solver examples/pretrained_caffemodel/bvlc_alexnet/bvlc_alexnet_solver_ucm.prototxt  -weights examples/pretrained_caffemodel/bvlc_alexnet_dsn_v1/bvlc_alexnet.caffemodel