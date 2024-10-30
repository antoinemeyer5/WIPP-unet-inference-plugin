# UNet CNN Semantic-Segmentation Inference plugin

## Docker distribution

This plugin is available on DockerHub from the WIPP organization

```
docker pull wipp/wipp-unet-cnn-inference-plugin
```

## Build Docker File
```bash
#!/bin/bash

version=1.0.0
docker build . -t wipp/wipp-unet-cnn-inference-plugin:latest
docker build . -t wipp/wipp-unet-cnn-inference-plugin:${version}
```

## Run Docker File

```bash
docker run  --gpus device=all \
    -v "path/to/input/data/folder":/data/inputs \
    -v "path/to/output/folder":/data/outputs \
    -v "path/to/model/folder":/data/model \
    wipp/wipp-unet-cnn-inference-plugin \
    --outputDir /data/outputs \
    --imageDir /data/inputs
    --savedModel /data/model 
```

## UNet Inference Job Options
```bash
usage: inference [-h] 
                --savedModel SAVED_MODEL_FILEPATH 
                --imageDir IMAGE_DIR
                --outputDir OUTPUT_DIR
                 [--useIntensityScaling USE_INTENSITY_SCALING] 
                 [--useTiling USE_TILING] 
                 [--tileSize TILE_SIZE] 


```
