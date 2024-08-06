#!/bin/bash
EXP_FILE="exps/default/yolox_m.py"
CKPT_PATH="pretrained_weight/hit-uav-weight/latest_ckpt.pth"
IMG_PATH="datasets/infer_images/hit_uav_test2017/"
CONF_THRESH=0.5
NMS_THRESH=0.45
IMG_SIZE=512
DEVICE="gpu"
python3 tools/demo.py image -f $EXP_FILE --trt --path $IMG_PATH --conf $CONF_THRESH --nms $NMS_THRESH --tsize $IMG_SIZE --save_result --device $DEVICE