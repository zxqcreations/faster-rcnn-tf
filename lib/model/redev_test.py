from redev import redev
import _init_paths

a = redev("/home/sook/code/faster-rcnn-tf/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_1.ckpt")

lst = a.get_variables_in_checkpoint_file()

for v, itm in lst.items():
    print(v, itm)
