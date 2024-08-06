#!/bin/bash
# BATCH_SIZE=8
# PRETRAINED_WEIGHT='./pretrained_weight/yolox_s.pth'
# python3 tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b $BATCH_SIZE --fp16 -o -c $PRETRAINED_WEIGHT

#!/bin/bash
BATCH_SIZE=8
PRETRAINED_WEIGHT='./pretrained_weight/yolox_m.pth'
python3 tools/train.py -f exps/example/custom/yolox_m.py -d 1 -b $BATCH_SIZE --fp16 -o -c $PRETRAINED_WEIGHT