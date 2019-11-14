import tensorflow as tf
import os
import pickle
import numpy as np

CIFAR_DIR = "e:/Program/Module/cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


def load_data(filename):

    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        return data['data'], data['labels']


class CifarData:

    def __init__(self,  filenames, need_shuffle):

        all_data = []
        all_labels = []

        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0

        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):

        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def size(self):
        return self._num_examples

    def next_batch(self, batch_size):

        if batch_size > self._num_examples:
            raise Exception("the batch size gt the size of all examples")

        end_indicator = self._indicator + batch_size

        if self._indicator < self._num_examples and end_indicator > self._num_examples:
            end_indicator = self._num_examples

        elif self._indicator >= self._num_examples:
            self._indicator = 0
            end_indicator = batch_size

        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator

        return batch_data, batch_labels


train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i)
                   for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)


def inception_block(x, output_channel_for_each_path, name):
    with tf.variable_scope(name):
        conv1_1 = tf.layers.conv2d(x, output_channel_for_each_path[0],
                                   (1, 1), strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv1_1')
        conv3_3 = tf.layers.conv2d(x, output_channel_for_each_path[1],
                                   (3, 3), strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv3_3')
        conv5_5 = tf.layers.conv2d(x, output_channel_for_each_path[2],
                                   (5, 5), strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv5_5')
        max_pooling = tf.layers.max_pooling2d(x, (2, 2), (2, 2),
                                              name='max_pooling')
    max_pooling_shape = max_pooling.get_shape().as_list()[1:]
    input_shape = x.get_shape().as_list()[1:]

    width_padding = (input_shape[0] - max_pooling_shape[0]) // 2
    height_padding = (input_shape[1] - max_pooling_shape[1]) // 2
    padding_pooling = tf.pad(max_pooling,
                             [
                                 [0, 0],
                                 [width_padding, width_padding],
                                 [height_padding, height_padding],
                                 [0, 0]
                             ])
    concat_layer = tf.concat(
        [conv1_1, conv3_3, conv5_5, padding_pooling], axis=3)
    return concat_layer


x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

x_image = tf.reshape(x, [-1, 3, 32, 32])
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

conv1 = tf.layers.conv2d(x_image, 32, (3, 3),
                         padding='same',
                         activation=tf.nn.relu,
                         name='conv1')
pooling1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), name='pool1')

inception_2a = inception_block(pooling1, [16, 16, 16], name='inception_2a')
inception_2b = inception_block(inception_2a, [16, 16, 16], name='inception_2b')

pooling2 = tf.layers.max_pooling2d(inception_2b, (2, 2), (2, 2), name='pool2')

inception_3a = inception_block(pooling2, [16, 16, 16], name='inception_3a')
inception_3b = inception_block(inception_3a, [16, 16, 16], name='inception_3b')

pooling3 = tf.layers.max_pooling2d(inception_3b, (2, 2), (2, 2), name='pool3')

flatten = tf.contrib.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
predict = tf.argmax(y_, 1)
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()

batch_size = 250
iteration = 200
epoch_num = 200
best_test_acc = 0

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_num):
        total_acc = 0.0
        total_loss = 0.0
        for i in range(1, iteration+1):
            batch_data, batch_labels = train_data.next_batch(batch_size)
            loss_val, acc_val, _ = sess.run([loss, accuracy, train_op], feed_dict={
                                            x: batch_data, y: batch_labels})
            total_acc += acc_val
            total_loss += loss_val
            print("Epoch: %d, iteration: %d/%d, [Train] loss: %4.5f, [Train] acc: %4.5f"
                  % (epoch, i, iteration, total_loss / i, total_acc / i), end='\r')
        print('\n')
        test_all_acc = []
        test_all_loss = []
        test_data = CifarData(test_filenames, False)
        test_size = test_data.size()
        test_batch_num = np.ceil(test_size / batch_size)
        for j in range(int(test_batch_num)):
            test_batch_data, test_batch_labels = test_data.next_batch(
                batch_size)
            test_acc, test_loss = sess.run([accuracy, loss], feed_dict={
                                           x: test_batch_data, y: test_batch_labels})
            test_all_acc.append(test_acc)
            test_all_loss.append(test_loss)

        if best_test_acc < np.mean(test_all_acc):
            best_test_acc = np.mean(test_all_acc)
            print("Epoch: %d, [Test] loss: %4.5f, [Test] acc: %4.5f Flag:*" %
                  (epoch, np.mean(test_all_loss), np.mean(test_all_acc)))
        else:
            print("[Test] loss: %4.5f, [Test] acc: %4.5f Flag:-" %
                  (epoch, np.mean(test_all_loss), np.mean(test_all_acc)))
