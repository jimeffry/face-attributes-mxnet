#! /bin/bash
#python im2rec.py --prefix FaceAnti --root /home/lxy/Downloads/DataSet/face_anti  --list True --recursive True --train-ratio 0.9  --test-ratio 0.1 \
 #               --no-shuffle True
python img2rec_v2.py --prefix ../../data/CelebA --root /data/Face_Reg/CelebA/img_detected  --center-crop False \
                    --num-thread 1 --resize 112