# -*- coding:utf-8 -*-
import scipy.io
import cPickle
import h5py

# input image dimensions
img_rows, img_cols = 28, 28

mat = scipy.io.loadmat('/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/NASA/SAT-4_SAT-6/sat-4-full.mat')

#print('mat', '\n', mat)
X_train = mat['train_x']
y_train = mat['train_y']
X_test = mat['test_x']
y_test = mat['test_y']
annotations = mat['annotations']

y_train = y_train.T
y_test = y_test.T

print 'X_train:', '\n', X_train.shape
print 'X_test:', '\n', X_test.shape
print 'y_train', '\n', y_train.shape
print 'y_test', '\n', y_test.shape
#print 'annotations', '\n', annotations


X_train = X_train.transpose(2,0,1,3)
X_test = X_test.transpose(2,0,1,3)
X_train = X_train[0:3]
X_test = X_test[0:3]
X_train = X_train.transpose(3,0,1,2)
X_test = X_test.transpose(3,0,1,2)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
#X_train /= 255
#X_test /= 255

#y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
#y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
y_train = y_train.astype("int32")
y_test = y_test.astype("int32")

print 'X_train:', '\n', X_train.shape
print 'X_test:', '\n', X_test.shape
print 'y_train', '\n', y_train.shape
print 'y_test', '\n', y_test.shape


'''
print('Building full SAT-4 dataset......')
output = open('SAT-4_full.pkl', 'wb')

cPickle.dump(X_train, output)
cPickle.dump(y_train, output)
cPickle.dump(X_test,  output)
cPickle.dump(y_test,  output)

output.close()
print 'save is done'
'''
#build small dataset

X_train_small = X_train[000000:-1]
X_test_small = X_test[0:-1]
y_train_small = y_train[000000:-1]
y_test_small = y_test[0:-1]



"""
print('Building small SAT-6 dataset......')
output = open('SAT-6_all_RGB.pkl', 'wb')

cPickle.dump(X_train_small, output)
cPickle.dump(y_train_small, output)
cPickle.dump(X_test_small,  output)
cPickle.dump(y_test_small,  output)

output.close()

print 'save is done'
"""
file_name = 'SAT-4_all_RGB.hdf5'
print(file_name)
f = h5py.File(file_name,'w')
f.create_dataset('X_train',data=X_train)
f.create_dataset('y_train',data=y_train)
f.create_dataset('X_test',data=X_test)
f.create_dataset('y_test',data=y_test)
f.close()

print 'save is done!'