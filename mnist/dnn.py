import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # due to tensorflow SEE error

import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None


def main(_):
    input_node_size = 784 # 28x28 사이즈의 mnist 입력 데이터
    hidden_node_size = 100
    output_node_size = 10

    # minst 데이터를 one_hot 인코딩 타입으로 가져 온다.
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


    X = tf.placeholder(tf.float32,[None,input_node_size], name="x_input")
    T = tf.placeholder(tf.float32, [None, output_node_size], name="target")

    # 가중치 초기화를 제비어로 초기화 한다.
    W1 = tf.get_variable("W1", shape=[input_node_size, hidden_node_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    # 바이어스 초기화를 0으로 한다.
    b1 = tf.get_variable(name="b1", shape=[hidden_node_size], initializer=tf.zeros_initializer())

    W2 = tf.get_variable("W2", shape=[hidden_node_size, hidden_node_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name="b2", shape=[hidden_node_size], initializer=tf.zeros_initializer())

    W3 = tf.get_variable("W3", shape=[hidden_node_size, output_node_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(name="b3", shape=[output_node_size], initializer=tf.zeros_initializer())

    # 드롭아웃 비율 (keep_prob)값을 placeholder 로 선언 한다.
    dropout_rate = tf.placeholder(tf.float32)
    # label(정답)을 placeholder로 선언 한다.
    T = tf.placeholder(tf.float32, [None, output_node_size], name="target")

    # tensorboard 디버깅을 위해 scope를 구별해서 나눈다.
    # 아래는 Fully connected layer 2개가 있는 딥러닝 구조 이다.
    # predict : input -> affine -> relu -> dropout -> affine -> relu -> dropout -> affine
    # loss : croos_entropy
    # optimizer : Adam

    with tf.name_scope("Layer2") as scope:
        L1_affine = tf.matmul(X, W1) + b1
        L1_activation = tf.nn.relu(L1_affine)
        L1_dropout = tf.nn.dropout(L1_activation, dropout_rate)

    with tf.name_scope("Layer3") as scope:
        L2_affine = tf.matmul(L1_dropout, W2) + b2
        L2_activation = tf.nn.relu(L2_affine)
        L2_dropout = tf.nn.dropout(L2_activation, dropout_rate)

    with tf.name_scope("Layer4") as scope:
        Y = tf.matmul(L2_dropout, W3) + b3

    with tf.name_scope("Loss") as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Y))
        loss_summ = tf.summary.scalar("Loss", cross_entropy)

    with tf.name_scope("Train") as scope:
        # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # accracy 방법은 tf.equal함수에 argmax 함수를 이용하여 가장 높은 값을 batch 된 갯수 만큼 True,False를 반환 받고
    # 이를 float32로 캐스팅하고 이값의 평균을 내어서 확률을 계산한다. (0.0 ~ 1.0)
    with tf.name_scope("Accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(T, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summ = tf.summary.scalar("Accuracy", accuracy, ["Epoch"])


    with tf.Session() as sess:
        with tf.device("/cpu:0"):  # device 별로 실행 시킬 수 있다.
            sess.run(tf.global_variables_initializer()) # tensorflw variables 초기화

            merged = tf.summary.merge_all()
            epoch_merged = tf.summary.merge_all("Epoch")
            writer = tf.summary.FileWriter("./logs/", sess.graph)

            batch_size = int(mnist.train.num_examples / 100)

            # Train
            for epoch in range(10): # 애폭 횟수 10회
                for step in range(batch_size):
                    image, label = mnist.train.next_batch(100)
                    summ, _ = sess.run([merged, train_step], feed_dict={X: image,
                                                                        T: label,
                                                                        dropout_rate: 0.7
                                                                   })
                    writer.add_summary(summ, epoch * batch_size + step)

                summ, _accuracy = sess.run([epoch_merged, accuracy],
                                           feed_dict={X: mnist.test.images,
                                                      T: mnist.test.labels,
                                                      dropout_rate: 1.0
                                                      })
                writer.add_summary(summ, (epoch + 1) * batch_size)

                print ("Epoch:", (epoch + 1))
                print ("Test Accuracy:", _accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run()

