import pandas as pd
import tensorflow as tf
import numpy as np
import os
import csv
import cv2

training_path = './train.csv'
testing_path = './test.csv'
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
DIGIT_RANGE = 10
batch_rate = 0.005

#training_df = pd.read_csv(training_path)
training_df = pd.read_csv(training_path)
testing_df = pd.read_csv(testing_path)


#training_set = np.array(training_df.iloc[:5000, 1:].astype('float32')) / 255
#training_labels = np.array(training_df.iloc[:5000, :1].astype('int'))
test_set = np.array(testing_df. iloc[:, :].astype('float32')) / 255
#print(training_set.shape)
#print(training_labels.shape)

#image = training_set

#labels = np.zeros((image.shape[0], DIGIT_RANGE))

#print(training_set)

#for i in range(0, image.shape[0]):
#    labels[i, training_labels[i]] = 1

def get_batch(reader):
    batch = reader.sample(frac=batch_rate)
    image = np.array(batch.iloc[:, 1:].astype('uint8'))
    training_labels = np.array(batch.iloc[:, :1].astype('int'))
    label = np.zeros((image.shape[0], DIGIT_RANGE))
    for i in range(0, image.shape[0]):
        label[i, training_labels[i]] = 1
        _img = np.reshape(image[i], (28, 28))
        ret3, th3 = cv2.threshold(_img, 0, 255, cv2.THRESH_OTSU)
        image[i] = np.reshape(th3, (784)).astype('float') / 255

    return image, label
#print(labels)

def weight_variable(name, shape):
    return(tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

def bias_variable(name, shape):
    return (tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

def con2d(x, W, _name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=_name)

def max_pooling_2x2(x, pool_height, pool_width, _name):
    # step change to 2
    layer = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=_name)
    pool_height = pool_height//2 + pool_height%2
    pool_width = pool_width//2 + pool_width%2
    return layer, pool_height, pool_width

g=tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None, IMAGE_WIDTH*IMAGE_HEIGHT]) # 28*28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    pool_heigth = IMAGE_HEIGHT
    pool_width = IMAGE_WIDTH
    with tf.device('/gpu:0'):
        W_conv1 = weight_variable('W1', [5, 5, 1, 16])
        b_conv1 = bias_variable('b1', [16])
        h_conv1 = tf.nn.relu(tf.nn.bias_add(con2d(x_image, W_conv1, "conv1"), b_conv1))
        h_pool1, pool_heigth, pool_width = max_pooling_2x2(h_conv1, pool_heigth, pool_width, "pool1")  # output 14 14
        h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

        W_conv2 = weight_variable('W2', [5, 5, 16, 32])
        b_conv2 = bias_variable('b2', [32])
        h_conv2 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool1, W_conv2, "conv2"), b_conv2))
        h_pool2, pool_heigth, pool_width = max_pooling_2x2(h_conv2, pool_heigth, pool_width, "pool2")  # output 7 7
        h_pool2 = tf.nn.dropout(h_pool2, keep_prob)
        #h_pool2_flat = tf.reshape(h_pool2, [-1, pool_heigth * pool_width * 32])
        
        W_conv3 = weight_variable('W3', [5, 5, 32, 32])
        b_conv3 = bias_variable('b3', [32])
        h_conv3 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool2, W_conv3, "conv3"), b_conv3))
        h_pool3, pool_heigth, pool_width = max_pooling_2x2(h_conv3, pool_heigth, pool_width, "pool3")  # output 4 4
        h_pool3 = tf.nn.dropout(h_pool3, keep_prob)

        W_conv4 = weight_variable('W4', [5, 5, 32, 16])
        b_conv4 = bias_variable('b4', [16])
        h_conv4 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool3, W_conv4, "conv4"), b_conv4))
        h_pool4, pool_heigth, pool_width = max_pooling_2x2(h_conv4, pool_heigth, pool_width, "pool4")  # output 2 2
        h_pool4 = tf.nn.dropout(h_pool4, keep_prob)

        W_conv5 = weight_variable('W5', [5, 5, 16, 10])
        b_conv5 = bias_variable('b5', [10])
        h_conv5 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool4, W_conv5, "conv5"), b_conv5))
        h_pool5, pool_heigth, pool_width = max_pooling_2x2(h_conv5, pool_heigth, pool_width, "pool5")  # output 2 2
        #h_pool5 = tf.nn.dropout(h_pool5, keep_prob)
        output = tf.reshape(h_pool5, [-1, 10])

        '''
	W_fc3 = weight_variable('W4', [pool_heigth * pool_width * 32, 512])
        b_fc3 = bias_variable('b4', [512])
        h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, W_fc3), b_fc3))
        h_fc3 = tf.nn.dropout(h_fc3, keep_prob)
        
        W_fc4 = weight_variable('W5', [512, DIGIT_RANGE])
        b_fc4 = bias_variable('b5', [DIGIT_RANGE])
        '''
        #output = tf.add(tf.matmul(h_fc3, W_fc4), b_fc4)
        prediction = tf.nn.softmax(output)
        
        #cross
        cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=output))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross)
        
        
        #accuracy
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        init = tf.global_variables_initializer()
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(init)
            tf.train.start_queue_runners(sess=sess)
            for i in range(10000):
                image, labels = get_batch(training_df)
                train_step.run(feed_dict={xs: image, ys: labels, keep_prob: 0.5})
                cross_sess = sess.run(cross, feed_dict={xs: image, ys: labels, keep_prob: 1})
                if i % 100 == 0:
                    print(cross_sess)
            test_predict = sess.run(output, feed_dict={xs: test_set, keep_prob: 1})
            test_labels = np.argmax(test_predict, axis=1)
            idx = np.array(range(1, test_labels.shape[0]+1))
            total = np.hstack((np.transpose([idx]), np.transpose([test_labels])))
            print(test_labels)
            with open("submission.csv", "w") as csvfile:
                writer = csv.writer(csvfile)
        
                # 先写入columns_name
                writer.writerow(["ImageId", "Label"])
                # 写入多行用writerows
                writer.writerows(total)
            coord.request_stop()
            coord.join(threads)

