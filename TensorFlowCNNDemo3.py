# coding=utf-8
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tempfile
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
    ## 第一层卷积操作 ##
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        variable_summaries(W_conv1)
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1)
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.summary.histogram('activations', h_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    ## 第二层卷积操作 ##
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2)
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        tf.summary.histogram('activations', h_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    ## 第三层全连接操作 ##
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        variable_summaries(W_fc1)
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        tf.summary.histogram('activations', h_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## 第四层输出操作 ##
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        variable_summaries(W_fc2)
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2)
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Import data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Create the model
    # 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
    x = tf.placeholder(tf.float32, [None, 784])

    # 类别是0-9总共10个类别，对应输出分类结果
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_conv, keep_prob = deepnn(x)



    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # setup recording variables
    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    # graph_location = tempfile.mkdtemp()
    # print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter('output/TensorFlowCNNDemo3/train')
    test_writer = tf.summary.FileWriter('output/TensorFlowCNNDemo3/test')
    train_writer.add_graph(tf.get_default_graph())



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3000):
            batch = mnist.train.next_batch(50)
            if i % 200 == 0:
                # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                summary, acc = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                test_writer.add_summary(summary, i)
                print('step %d, training accuracy %g' % (i, acc))
            else:
                # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                train_writer.add_summary(summary, i)
                # print('step %d, training accuracy %g' % (i, _))
        # print('test accuracy %g' % accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        train_writer.add_graph(sess.graph)

        saver = tf.train.Saver()
        path = saver.save(sess, 'model/TensorFlowCNNDemo3.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)