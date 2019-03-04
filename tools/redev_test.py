import _init_paths
from redev import redev

a = redev("/home/sook/code/faster-rcnn-tf/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_1.ckpt")

#lst = a.get_variables_in_checkpoint_file()

a.create_architecture("TEST", 8,
                            tag='default', anchor_scales=[8, 16, 32])
a.restore_and_redev()
#for v, itm in lst.items():
#    print(v, itm)
