#!/bin/bash
WEIGHT_PATH=../ros_ws/src/ros2-detection-python/weights/yolox_weights/yolox_m.pth
python3 tools/export_onnx.py --output-name yolox_m.onnx -f exps/default/yolox_m.py -c $WEIGHT_PATH