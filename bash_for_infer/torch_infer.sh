#!/bin/bash
EXP_FILE="exps/default/yolox_m.py"
CKPT_PATH="pretrained_weight/flir_cam_200eph_yoloxm/yolox_m/latest_ckpt.pth"
IMG_PATH="datasets/infer_images/flair_cam_obj_test/"
CONF_THRESH=0.5
NMS_THRESH=0.45
IMG_SIZE=512
DEVICE="gpu"
python3 tools/demo.py image -f $EXP_FILE -c $CKPT_PATH --path $IMG_PATH --conf $CONF_THRESH --nms $NMS_THRESH --tsize $IMG_SIZE --save_result --device $DEVICE