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

        print(self._data.shape)
        print(self._labels.shape)

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


def conv2d(input, name, k_size):
    return tf.layers.conv2d(input, k_size, (3, 3), strides=(1, 1), padding='same',
                            activation=None, name=name)


def max_pool(input, name):
    return tf.layers.max_pooling2d(input, (2, 2), (2, 2), name=name)


def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


def model():

    x = tf.placeholder(tf.float32, [None, 3072])
    y = tf.placeholder(tf.int64, [None])

    x_image = tf.reshape(x, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

    conv1_1 = tf.nn.relu(batch_norm(conv2d(x_image, 'conv1_1', 64)))
    conv1_2 = tf.nn.relu(batch_norm(conv2d(conv1_1, 'conv1_2', 64)))

    pooling1 = max_pool(conv1_2, 'pool1')

    conv2_1 = tf.nn.relu(batch_norm(conv2d(pooling1, 'conv2_1', 128)))
    conv2_2 = tf.nn.relu(batch_norm(conv2d(conv2_1, 'conv2_2', 128)))

    pooling2 = max_pool(conv2_2, 'pool2')

    conv3_1 = tf.nn.relu(batch_norm(conv2d(pooling2, 'conv3_1', 256)))
    conv3_2 = tf.nn.relu(batch_norm(conv2d(conv3_1, 'conv3_2', 256)))
    conv3_3 = tf.nn.relu(batch_norm(conv2d(conv3_2, 'conv3_3', 256)))
    conv3_4 = tf.nn.relu(batch_norm(conv2d(conv3_3, 'conv3_4', 256)))

    pooling3 = max_pool(conv3_4, 'pool3')

    conv4_1 = tf.nn.relu(batch_norm(conv2d(pooling3, 'conv4_1', 512)))
    conv4_2 = tf.nn.relu(batch_norm(conv2d(conv4_1, 'conv4_2', 512)))
    conv4_3 = tf.nn.relu(batch_norm(conv2d(conv4_2, 'conv4_3', 512)))
    conv4_4 = tf.nn.relu(batch_norm(conv2d(conv4_3, 'conv4_4', 512)))

    pooling4 = max_pool(conv4_4, 'pool4')

    conv5_1 = tf.nn.relu(batch_norm(conv2d(pooling4, 'conv5_1', 512)))
    conv5_2 = tf.nn.relu(batch_norm(conv2d(conv5_1, 'conv5_2', 512)))
    conv5_3 = tf.nn.relu(batch_norm(conv2d(conv5_2, 'conv5_3', 512)))
    conv5_4 = tf.nn.relu(batch_norm(conv2d(conv5_3, 'conv5_4', 512)))

    flatten = tf.contrib.layers.flatten(conv5_4)

    fc1 = tf.nn.relu(batch_norm(tf.layers.dense(flatten, 4096)))
    fc2 = tf.nn.relu(batch_norm(tf.layers.dense(fc1, 4096)))
    fc3 = tf.nn.relu(batch_norm(tf.layers.dense(fc2, 10)))

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=fc3)
    predict = tf.argmax(fc3, 1)
    correct_prediction = tf.equal(predict, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    return x, y, loss, accuracy, train_op


train_flag = tf.placeholder(tf.bool)
batch_size = 250
num_epoch = 164

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i)
                   for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)
train_size = train_data.size()
batch_data, batch_labels = train_data.next_batch(batch_size)

x, y, loss, accuracy, train_op = model()
best_test_acc = 0
with tf.Session() as sess:
    init = tf.global_variables_initializer().run()
    iters = 0
    for epoch in range(num_epoch):
        batch_num = np.ceil(train_size / batch_size)
        total_acc = 0.0
        total_loss = 0.0
        for i in range(1, int(batch_num)+1):
            batch_data, batch_labels = train_data.next_batch(batch_size)
            loss_val, acc_val, _ = sess.run([loss, accuracy, train_op], feed_dict={
                                            x: batch_data, y: batch_labels, train_flag: True})
            total_acc += acc_val
            total_loss += loss_val
            print("Epoch: %d, iteration: %d/%d, [Train] loss: %4.5f, [Train] acc: %4.5f"
                          % (epoch, i, batch_num, total_loss / i, total_acc / i), end='\r')
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
                                           x: test_batch_data, y: test_batch_labels, train_flag: False})
            test_all_acc.append(test_acc)
            test_all_loss.append(test_loss)

        if best_test_acc < np.mean(test_all_acc):
            best_test_acc = np.mean(test_all_acc)
            print("Epoch: %d, [Test] loss: %4.5f, [Test] acc: %4.5f Flag:*" %
                  (epoch, np.mean(test_all_loss), np.mean(test_all_acc)))
        else:
            print("Epoch: %d, [Test] loss: %4.5f, [Test] acc: %4.5f Flag:-" %
                  (epoch, np.mean(test_all_loss), np.mean(test_all_acc)))
