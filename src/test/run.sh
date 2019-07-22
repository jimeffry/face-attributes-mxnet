#! /bin/bash
#python demo.py --img-path1 test.jpg --gpu 0 --load-epoch 10 --cmd-type imgtest

##test filelist
#python demo.py --file-in ../../data/FaceAnti_test.lst --out-file ./output/record4.txt --base-dir /home/lxy/Downloads/DataSet/face_anti \
 #       --load-epoch 25 --cmd-type txtlisttest

#python test.py  --load-epoch 15 --cmd-type video
python val.py --file-in ../../data/CelebA/test.lst --out-file ./output/attribute_mobilenet_rgbbar.txt --base-dir /data/Face_Reg/CelebA/img_detected \
        --load-epoch 30 --cmd-type evalue
#python plot.py --file-in ./output/record_test.txt --base-name test_data --high 1000 --unit 100 --cmd-type plot2data
python plot.py --file-in ./output/attribute_mobilenet_rgbbar.txt --base-name mobilenet_rgb --cmd-type plot3data
