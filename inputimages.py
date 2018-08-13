#https://github.com/kalaspuffar/tensorflow-data/blob/master/create_dataset.py
from random import shuffle
import random
import glob
import sys
import cv2
import os
import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def salt_pepper(img, prob):
    
    #A simple one that randomize
    #Add the sale and pepper to the image, according to the probability.
    output = np.zeros(img.shape, np.uint8)
    threshold =  1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rand = random.random()
            if rand < prob:
                output[i][j] = 0
            elif rand > threshold:
                output[i][j] = 255
            else: output[i][j] = img[i][j]
    return output



def data_preprocessing(addrs, labels):
    
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        img = load_image(addrs[i])

        img_name, extension = os.path.splitext(os.path.basename(addrs[i]))
        path_name = os.path.dirname(addrs[i])
        print(img_name)
        print(path_name)
        

        #rotate image
        for j in range(3):
            if j != 1:
                rotate = np.rot90(img, k=1+j)
                cv2.imwrite(os.path.join(path_name, img_name +'_R' + str(j) + '.jpg' ), rotate)

        #salt and pepper
        s_p = salt_pepper(img, prob = 0.005)
        cv2.imwrite(os.path.join(path_name, img_name +'_SP' + '.jpg' ), rotate)
    
def createDataRecord(out_filename, addrs, labels, mode):
    # open the TFRecords file
    prob = 0.005
    imageCounter = 0
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])

        #img_name, extension = os.path.splitext(addrs[i])
        img_name, extension = os.path.splitext(os.path.basename(addrs[i]))
        path_name = os.path.dirname(addrs[i])
        print(path_name)
        #print(os.path.dirname(addrs[i]))
        #print(img_name)

        if mode == 'train':
            #rotate image
            for j in range(3):
                if j != 1:
                    rotate = np.rot90(img, k=1+j)
                    cv2.imwrite(os.path.join(path_name, img_name +'_R' + str(j) + '.jpg' ), rotate)
                    label = labels[i]
                    if rotate is None:
                        continue
                     # Create a feature
                    feature = {
                        'image_raw': _bytes_feature(rotate.tostring()),
                        'label': _int64_feature(label)
                    }
                    imageCounter += 1
                    
                    # Create an example protocol buffer
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    
                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())

##            output = np.zeros(img.shape, np.uint8)
##            threshold =  1 - prob
##            for i in range(img.shape[0]):
##                for j in range(img.shape[1]):
##                    rand = random.random()
##                    if rand < prob:
##                        output[i][j] = 0
##                    elif rand > threshold:
##                        output[i][j] = 255
##                    else:
##                        output[i][j] = img[i][j]
                    
            s_p = salt_pepper(img, prob)

            cv2.imwrite(os.path.join(path_name, img_name +'_SP' + '.jpg' ), s_p)

            #print(s_p)
            label = labels[i]
                        
            if s_p is None:
                continue
            imageCounter += 1
            
            #Create a feature
            feature = {
                'image_raw': _bytes_feature(s_p.tostring()),
                'label': _int64_feature(label)
            }

            #Create an example protocol buffer
            example =  tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            
            
        
        label = labels[i]

        if img is None:
            continue
        imageCounter += 1
        # Create a feature
        
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        #print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    print(mode + str(" : ") + str(imageCounter))
        
    writer.close()
    sys.stdout.flush()

imageclef_train_path = 'ImageCLEF2013/*/*.jpg'
# read addresses and labels from the 'train' folder
#Return a possibly-empty list of path names that match pathname
addrs = glob.glob(imageclef_train_path) 
labels = [0 if 'D3DR' in addr else
          1 if 'DMEL' in addr else
          2 if 'DMFL' in addr else
          3 if 'DMLI' in addr else
          4 if 'DMTR' in addr else
          5 if 'DRAN' in addr else
          6 if 'DRCO' in addr else
          7 if 'DRCT' in addr else
          8 if 'DRMR' in addr else
          9 if 'DRPE' in addr else
          10 if 'DRUS' in addr else
          11 if 'DRXR' in addr else
          12 if 'DSEC' in addr else
          13 if 'DSEE' in addr else
          14 if 'DSEM' in addr else
          15 if 'DVDM' in addr else
          16 if 'DVEN' in addr else
          17 if 'DVOR' in addr else
          18 if 'GCHE' in addr else
          19 if 'GFIG' in addr else
          20 if 'GFLO' in addr else
          21 if 'GGEL' in addr else
          22 if 'GGEN' in addr else
          23 if 'GHDR' in addr else
          24 if 'GMAT' in addr else
          25 if 'GNCP' in addr else
          26 if 'GPLI' in addr else
          27 if 'GSCR' in addr else
          28 if 'GSYS' in addr else
          #29 if 'GTAB' in addr
          29 for addr in addrs]  # 0 = Cat, 1 = Dog

# to shuffle data
c = list(zip(addrs, labels))
shuffle(c)
#print(c)
addrs, labels = zip(*c)

print(len(addrs))
print(len(labels))

data_preprocessing(addrs, labels)

### Divide the data into 70% train, 30% test
##train_addrs = addrs[0:int(0.7*len(addrs))]
##train_labels = labels[0:int(0.7*len(labels))]
###val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
###val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
##test_addrs = addrs[int(0.7*len(addrs)):]
##test_labels = labels[int(0.7*len(labels)):]
##
##createDataRecord('train.tfrecords', train_addrs, train_labels, 'train')
###createDataRecord('val.tfrecords', val_addrs, val_labels)
##createDataRecord('test.tfrecords', test_addrs, test_labels, 'test')
    
