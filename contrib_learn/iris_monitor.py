import tensorflow as tf
import os
import numpy as np
import urllib.request



os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # due to tensorflow SEE error


tf.logging.set_verbosity(tf.logging.INFO)


# Data sets
IRIS_TRAINING = os.path.join(os.path.dirname(__file__), "iris_training.csv")
IRIS_TEST = os.path.join(os.path.dirname(__file__), "iris_test.csv")

def main(unused_argv):
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float)

    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key="classes"),
        "precision":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key="classes"),
        "recall":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_recall,
                prediction_key="classes")
    }

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_set.data,
        test_set.target,
        every_n_steps=50,
        metrics=validation_metrics,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True, # early_stopping_rounds 지정된 step 동안 early_stopping_metric로 지정된 값이 변화가 없으면 트레이닝을 중지한다.
        early_stopping_rounds=200)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir="/tmp/iris_model",
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1)) # 데이터의

    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

    classifier.fit(input_fn=get_train_inputs,
        steps=2000,
        monitors=[validation_monitor])

    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)

        return x, y

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(
        input_fn=get_test_inputs)["accuracy"]
    print("Accuracy: {0:f}".format(accuracy_score))

    # Classify two new flower samples.
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
    y = list(classifier.predict(new_samples))
    print("Predictions: {}".format(str(y)))

if __name__ == "__main__":
  tf.app.run()