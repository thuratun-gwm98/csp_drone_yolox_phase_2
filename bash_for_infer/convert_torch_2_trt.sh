YOLOX_MODEL_NAME="yolox-m"
EXP_FILE="./exps/default/yolox_m.py"
YOLOX_CKPT='pretrained_weight/hit-uav-weight/latest_ckpt.pth'
BATCH_SIZE=1
python3 tools/trt.py --exp_file $EXP_FILE -c $YOLOX_CKPT -b $BATCH_SIZE