# -*- coding:utf8 -*-

from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from activation.PL_ReLU import *
from data_preprocessing.data_load import load_data_split_pickle
from matplotlib.pylab import *
import os
import csv


batch_size = 128
nb_classes = 7
nb_epoch = 2000
data_augmentation = True
# shape of the image (SHAPE x SHAPE)
img_rows, img_cols = 64, 64
# the CIFAR10 images are RGB
img_channels = 3


# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = load_data_split_pickle(dataset=('/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/data_preprocessing/testvec_64'\
                                ,'/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/data_preprocessing/trainvec_64'))
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#building CNN model
print('Building model......')
print('64x64 b128 1:1 lr=0.005, decay=1e-6, momentum=0.9 relu rotate30 s0.3 e2000')
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(img_channels, img_rows, img_cols)))
model.add(Lrelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Convolution2D(64, 3, 3))
model.add(Lrelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(512))
model.add(Lrelu())
model.add(Dropout(0.5))
#sofmax分类
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
#载入使用相同网络结构训练的NASA数据集的权重
## not work  Exception: Layer shape (10816, 512) not compatible with weight shape (1024, 512).

#model.load_weights('/home/dell/exs/NASA/log_records/6classRGB.hdf5')
# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

if not data_augmentation:
    print("Not using data augmentation or normalization")
    print('using LRelu activation')

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,show_accuracy=True,validation_data=(X_test,Y_test))


else:
    print("Using real time data augmentation")
    testAccu = []
    trainAccu = []
    testLoss = []
    trainLoss = []
    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        train_flag = 0
        trainAcAll = 0
        trainLoAll = 0
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train,batch_size = batch_size):
            train_flag += 1
            trainLo , trainAc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            trainAcAll += trainAc
            trainLoAll += trainLo
            progbar.add(X_batch.shape[0], values=[("train loss", trainLo), ("train accuracy:", trainAc)])

        print("Testing...")
        # test time!
        testLo,testAc = model.evaluate(X_test,Y_test,show_accuracy=True)
        print ('test loss: ' + str(testLo) + '    ' + 'test Accuracy: ' + str(testAc))
        testAccu.append(testAc)
        trainAccu.append(trainAcAll/train_flag)

        testLoss.append(testLo)
        trainLoss.append(trainLoAll/train_flag)
    Pred = model.predict_classes(X_test)

# draw acc loss figure
    p_testAcc, = plt.plot(testAccu, color='blue')
    p_trainAcc, = plt.plot(trainAccu, color='red')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend([p_testAcc, p_trainAcc], ('testAcc', 'trainAcc'), 'best', numpoints=1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/log/ConvNet_64/RSSCN7_RGB_r30_3_Accu.eps', dpi=128)
#    plt.show()
    plt.figure()
    p_testLoss, = plt.plot(testLoss, color='blue')
    p_trainLoss, = plt.plot(trainLoss, color='red')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend([p_testLoss, p_trainLoss], ('testLoss', 'trainLoss'), 'best', numpoints=1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/log/ConvNet_64/RSSCN7_RGB_r30_3_Loss.eps', dpi=128)
    print("figure saved")
# save weights
    model.save_weights('/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/log/ConvNet_64/RSSCN7_RGB_r30_3.hdf5',overwrite=True)
    print("weights saved")


# save class_name.txt and labels.txt to draw confusion matrix


    labelFile = '/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/log/ConvNet_64/labels_RSSCN7_r30_3.txt'
    f1 = open(labelFile,'w')
    print (Pred)
    print ("Pred length: " + str(len(Pred)))

    for i in range(len(Pred)):
        line = str(y_test[i]) + ' ' + str(Pred[i]) + '\n'
        f1.write(line)
    f1.close()

    class_name = '/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/Rui/General_RSSCN7/log/ConvNet_64/class_name_RSSCN7.txt'
    f2 = open(class_name, 'w')
    names = []
    for img in os.listdir('/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/RSSCN7_res'):# need to be changed
        names.append(img)
    print ("names Length:" + str(len(names)))
    for i in range(len(names)):
        line = str(names[i]) +  '\n'
        f2.write(line)
    f2.close()

"""
    Pred = model.predict_classes(X_test)

    plt.plot(testAccu,color = 'blue')
    plt.plot(trainAccu,color = 'red')
    plt.savefig('/home/dell/exs/NASA/log_records/6classRGBAccu.png', dpi=128)
    plt.figure()
    plt.plot(testLoss,color = 'blue')
    plt.plot(trainLoss,color = 'red')
    plt.savefig('/home/dell/exs/NASA/log_records/6classRGBLoss.png', dpi=128)
    model.save_weights('/home/dell/exs/NASA/log_records/6classRGB.hdf5',overwrite=True)


    Pred = model.predict_classes(X_test)

    labelFile = 'labels.txt'
    f1 = open(labelFile,'w')
    print (Pred)
    print ("Pred length: " + str(len(Pred)))

    for i in range(len(Pred)):
        line = str(y_test[i]) + ' ' + str(Pred[i]) + '\n'
        f1.write(line)
    f1.close()

    class_name = 'class_name.txt'
    f2 = open(class_name, 'w')
    names = []
    for img in os.listdir('/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/RSSCN7/RSSCN7_res'):
        names.append(img)
    print ("names Length:" + str(len(names)))
    for i in range(len(names)):
        line = str(names[i]) +  '\n'
        f2.write(line)
    f2.close()
"""