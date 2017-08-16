from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read().decode()
        with open(IRIS_TRAINING, 'w') as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read().decode()
        with open(IRIS_TEST, 'w') as f:
            f.write(raw)

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int, # 타겟(정답)의 데이터형
        features_dtype=np.float32) # 특성의 데이터형
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    # 연속적인 컬럼 데이터 설정 (꽃받침 길이,꽃받침 너비, 꽃잎 길이, 꽃잎 너비)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3, # 타겟 클래스 노드 갯수
                                                model_dir="/tmp/iris_model") # 훈련 중 체크포인터를 저장할 디렉토리
    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)

        return x, y

    # 학습 시키기
    classifier.fit(input_fn=get_train_inputs, steps=2000)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)

        return x, y

    # 모델의 정확도 판단하기
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # 새로운 두 꽃의 표본을 분류합니다.
    def new_samples():
        return np.array(
            [[6.4, 3.2, 4.5, 1.5],
             [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))

    print(
        "New Samples, Class Predictions:    {}\n"
            .format(predictions))

if __name__ == "__main__":
    main()