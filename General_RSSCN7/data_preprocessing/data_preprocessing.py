# -*- coding:utf-8 -*-

import os
import sys
import random
import PIL.Image as Image
import numpy as np


def resize_imgs(src_folder,res_folder):
    flag = 0
    for sub_folder in os.listdir(src_folder):
        img_folder = src_folder +'/'+ sub_folder
        for img_file in os.listdir(img_folder):
            im_path = src_folder + "/" + img_file
            im = Image.open(im_path)
            im_resize = im.resize((28,28))
            im_resize_path = res_folder + '/' + img_file
            im_resize.save(im_resize_path)
            flag += 1
            print flag

def reConstruct_folder(src_folder,res_folder,res_size):
    flag=1
    for src_img in os.listdir(src_folder):
        src_img_path = src_folder + "/" + src_img
        res_img_path = res_folder + "/" + src_img
        im = Image.open(src_img_path)
        new_im = im.resize(res_size)
        new_im.save(res_img_path)
        flag +=1
        if flag%100 == 0:
            print "num:" + str(flag)

def split_imgs(src_folder):
    test_set  = []
    train_set = []

    label = 0
    flag = 1
    for tea_folder in os.listdir(src_folder):
        tea_path = src_folder +'/'+ tea_folder
        tea_imgs = []
        for img_file in os.listdir(tea_path):
            img_path = tea_path +'/'+ img_file
            im = Image.open(img_path)
            x_size, y_size = im.size
            if x_size==28 and y_size==28:
                tea_imgs.append((img_path, label))
                flag += 1
        random.shuffle(tea_imgs)
        if tea_folder=='aGrass':
            train_set += tea_imgs[0:200]
            test_set  += tea_imgs[200:]
        elif tea_folder=='bField':
            train_set += tea_imgs[0:200]
            test_set  +=tea_imgs[200:]
        elif tea_folder=='cIndustry':
            train_set += tea_imgs[0:200]
            test_set  += tea_imgs[200:]
        elif tea_folder=='dRiverLake':
            train_set += tea_imgs[0:200]
            test_set  +=tea_imgs[200:]
        elif tea_folder=='eForest':
            train_set += tea_imgs[0:200]
            test_set  += tea_imgs[200:]
        elif tea_folder=='fResident':
            train_set += tea_imgs[0:200]
            test_set  +=tea_imgs[200:]
        elif tea_folder=='gParking':
            train_set += tea_imgs[0:200]
            test_set  += tea_imgs[200:]
        random.shuffle(train_set)
        random.shuffle(test_set)
        sys.stdout.flush()
        label += 1
    print 'all images:' + ''+ str(flag)
    print 'dataset num: %d' % (len(tea_imgs))
    print 'test  set num: %d' % (len(test_set))
    print 'train set num: %d' % (len(train_set))
    return test_set, train_set

def set_to_csv_file(data_set, file_name):
    f = open(file_name, 'wb')
    for item in data_set:
        line = item[0] + ',' + str(item[1]) + '\n'
        f.write(line)
    f.close()

def read_csv_file(csv_file):
    path_and_labels = []
    f = open(csv_file, 'rb')
    for line in f:
        line = line.strip('\r\n')
        path, label = line.split(',')
        label = int(label)
        path_and_labels.append((path, label))
    f.close()
    random.shuffle(path_and_labels)
    return path_and_labels

def vec_imgs(path_and_labels, image_size):
    image_vector_len = np.prod(image_size)

    arrs   = []
    labels = []
    i = 0
    for path_and_label in path_and_labels:
        path, label = path_and_label
        img = Image.open(path)
        arr_img = np.asarray(img, dtype='string')
        arr_img = arr_img.transpose(2,0,1)

        labels.append(label)
        arrs.append(arr_img)

        i += 1
        if i % 100 == 0:
            sys.stdout.write('\rdone: ' + str(i))
            sys.stdout.flush()
    print ''
    arrs = np.asarray(arrs, dtype='float32')
    labels = np.asarray(labels, dtype='int32')
    return (arrs, labels)

def cPickle_output(vars, file_name):
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def output_data(vector_vars, vector_folder, batch_size=1000):
    if not vector_folder.endswith('/'):
        vector_folder += '/'
    if not os.path.exists(vector_folder):
        os.mkdir(vector_folder)
    x, y = vector_vars
    n_batch = len(x) / batch_size
    for i in range(n_batch):
        file_name = vector_folder + str(i) + '.pkl'
        batch_x = x[ i*batch_size: (i+1)*batch_size]
        batch_y = y[ i*batch_size: (i+1)*batch_size]
        cPickle_output((batch_x, batch_y), file_name)
    if n_batch * batch_size < len(x):
        batch_x = x[n_batch*batch_size: ]
        batch_y = y[n_batch*batch_size: ]
        file_name = vector_folder + str(n_batch) + '.pkl'
        cPickle_output((batch_x, batch_y), file_name)


#resize image


res_folder = '/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/RSSCN7_res'
src_folder = '/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/RSSCN7_src'
for sub_src_folder in os.listdir(src_folder):
    img_src_path = src_folder + '/' + sub_src_folder
    img_res_path = res_folder + '/' + sub_src_folder
    if not os.path.exists(img_res_path):
        os.mkdir(img_res_path)
    print img_src_path
    print img_res_path
    reConstruct_folder(src_folder=img_src_path, res_folder=img_res_path, res_size=(28, 28))
"""
"""
"""
#generate dataset list
"""
test_set_file  = "/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/datafile/testfile_28"
train_set_file = "/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/datafile/trainfile_28"
test_set, train_set = split_imgs("/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/RSSCN7_res")
set_to_csv_file(test_set,  test_set_file)
set_to_csv_file(train_set, train_set_file)


"""
#vec_imgs
"""
test_path_and_labels  = read_csv_file("/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/datafile/testfile_28")
train_path_and_labels = read_csv_file("/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/datafile/trainfile_28")
print 'test  img num: %d' % (len(test_path_and_labels))
print 'train img num: %d' % (len(train_path_and_labels))
img_size = (3, 64, 64)  # channel, height, width
test_vec  = vec_imgs(test_path_and_labels, img_size)
train_vec = vec_imgs(train_path_and_labels, img_size)
output_data(test_vec,  '/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/data_preprocessing/testvec_28')
output_data(train_vec, '/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/data_preprocessing/trainvec_28')