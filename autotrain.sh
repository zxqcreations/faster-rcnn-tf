#!/bin/sh

# scp -r test@10.12.34.57:~/code/testcode/output ~/code/upload/
# cp ~/code/upload/output/* ~/code/faster-rcnn-tf/data/imagenet_weights/

if [ -d "./data/cache" ]; then
	echo "removing ./data/cache"
	rm -rf ./data/cache
	echo "done!"
fi
if [ -d "./output" ]; then
	echo "removing ./output"
	rm -rf ./output
	echo "done!"
fi
if [ -d "./tensorboard" ]; then
	echo "removing ./tensorboard"
	rm -rf ./tensorboard
	echo "done"
fi
if [ -f "vgg16_frcnn_frozen.pb" ]; then
	echo "removing ./vgg16_frcnn_frozen.pb"
	rm -r vgg16_frcnn_frozen.pb
        echo "done"
fi

if [ -d "./new_ckpt" ]; then
        echo "removing ./new_ckpt"
        rm -rf ./new_ckpt
        echo "done"
fi

echo "#######################################################\n"
echo "---------------- start training! ----------------------\n"
echo "#######################################################\n"

./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16

echo "#######################################################\n"
echo "---------------- training done! -----------------------\n"
echo "#######################################################\n"

#python3 ./tools/redev_test.py


#echo "\n\n now freezing model\n\n"

#python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py --input_graph=vgg16_faster_rcnn.pb --input_binary=true --input_checkpoint=new_ckpt/vgg16_frcnn_new.ckpt --output_graph=vgg16_frcnn_frozen.pb --output_node_name=concat/concat

#echo "\n\n frozen! now upload to 10.12.34.57\n\n"

#scp vgg16_frcnn_frozen.pb test@10.12.34.57:/home/code

#echo "All Done!!!"
