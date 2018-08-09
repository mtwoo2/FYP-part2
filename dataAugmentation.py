import tensorflow as tf
import numpy as np
import cv2
import sys
import glob
#import matplotlib.pylot as plt
#tf.enable_eager_execution()

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, shape=[224, 224, 3])
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label

data_path = 'train.tfrecords' #address to save the hdf5 file
dataset = tf.data.TFRecordDataset(data_path)
#dataset.apply(tf.contrib.data.shuffle_and_repeat(2014,1))
dataset = dataset.map(parser)
IMAGE_SIZE = 299
print(dataset)

#https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
"""def rotate_images(img):
    X_rotate = []
    cv2.imshow('image', img)
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #for img in X_imgs:
        #cv2.imshow('image', img)
        for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
                cv2.imshow('image', rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    print(len(X_rotate))
    
    #cv2.imshow('rotate', X_rotate[0])
    return X_rotate"""
    
def rotate_images(img):
        cv2.imshow('image', img)
        #for img in X_imgs:
        #https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
        # Placeholders: 'x' = A single image, 'y' = A batch of images
        # 'k' denotes the number of 90 degree anticlockwise rotations
        shape = [299, 299, 3]
        x = tf.placeholder(dtype = tf.float32, shape = shape)
        #k = tf.placeholder(tf.int32)
        rot_90 = tf.image.rot90(img, k=1)
        rot_180 = tf.image.rot90(img, k=2)
        print(rot_90)
        cv2.imshow('image', rot_90)
        # To rotate in any angle. In the example below, 'angles' is in radians
        #shape = [batch, height, width, 3]
        #y = tf.placeholder(dtype = tf.float32, shape = shape)
        #rot_tf_180 = tf.contrib.image.rotate(y, angles=3.1415)
        # Scikit-Image. 'angle' = Degrees. 'img' = Input Image
        # For details about 'mode', checkout the interpolation section below.
        #rot = skimage.transform.rotate(img, angle=45, mode='reflect')    
    

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

addr = 'dog.jpg'
rotated_imgs = rotate_images(load_image(addr))
