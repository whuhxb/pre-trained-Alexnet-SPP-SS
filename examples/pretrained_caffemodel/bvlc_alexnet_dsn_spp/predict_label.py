#coding=utf-8  
      
import os
import caffe  
import numpy as np  
root='/home/stu_3/Documents/caffe/'  
deploy=root + 'examples/pretrained_caffemodel/bvlc_alexnet_dsn_spp/deploy_rs19.prototxt'     
caffe_model=root + 'examples/pretrained_caffemodel/bvlc_alexnet_dsn_spp/output-RS19-1/bvlc_alexnet_train_RS19_iter_150000.caffemodel'   
#deploy=root + 'examples/pretrained_caffemodel/bvlc_alexnet_dsn_spp/deploy.prototxt' 
#caffe_model=root + 'examples/pretrained_caffemodel/bvlc_alexnet_dsn_spp/bvlc_alexnet.caffemodel'
import os
dir = root+'data/myself-RS19/22172/val/Airport/'
filelist=[]
filenames = os.listdir(dir)
for fn in filenames:
   fullfilename = os.path.join(dir,fn)
   filelist.append(fullfilename)

img=root+'data/myself-RS19/22172/val/Airport/airport_02.jpg'

def Test(img):
    net = caffe.Net(deploy,caffe_model,caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    #transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    im=caffe.io.load_image(img)
    net.blobs['data'].data[...] = transformer.preprocess('data',im)
    out = net.forward()

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    print labels
    prob= net.blobs['prob'].data[0].flatten()
    print prob
    order=prob.argsort()[999]
    #argsort()
    print 'the class is:',labels[order]
    f=file('/home/stu_3/Documents/caffe/data/myself-RS19/22172/pred_val.txt','a+')
    f.writelines(img+' '+labels[order]+'\n')
##
labels_filename = root +'data/myself-RS19/22172/DR.txt'
##
for i in range(0, len(filelist)):
    img= filelist[i]
    Test(img)

