#! /bin/bash
#python demo.py --img-path1 test.jpg --gpu 0 --load-epoch 10 --cmd-type imgtest

##test filelist
python demo.py --file-in ../../data/FaceAnti_test.lst --out-file ./output/record4.txt --base-dir /home/lxy/Downloads/DataSet/face_anti \
        --load-epoch 22 --cmd-type filetest