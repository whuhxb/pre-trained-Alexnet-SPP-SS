#!/usr/bin/env sh
export PYTHONPATH=$PWD/python:$PWD/examples/pretrained_caffemodel/bvlc_alexnet_spp

./build/tools/caffe train -solver examples/pretrained_caffemodel/bvlc_alexnet_spp/bvlc_alexnet_spp_solver_google.prototxt  -weights examples/pretrained_caffemodel/bvlc_alexnet_dsn_v1/bvlc_alexnet.caffemodel