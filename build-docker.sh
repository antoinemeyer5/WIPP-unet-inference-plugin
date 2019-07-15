#!/bin/bash

version=0.0.1
#docker build . -t wipp/wipp-unet-cnn-inference-plugin:latest
docker build . -t wipp/wipp-unet-cnn-inference-plugin:${version}
