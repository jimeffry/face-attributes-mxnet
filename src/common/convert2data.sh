#! /bin/bash
python im2rec.py --prefix FaceAnti --root /home/lxy/Downloads/DataSet/face_anti  --list True --recursive True --train-ratio 0.9  --test-ratio 0.1 \
                --no-shuffle True
#python im2rec.py --prefix FaceAnti --root /home/lxy/Downloads/DataSet/face_anti  --pass-through False --center-crop True \
 #                   --num-thread 1 --resize 224