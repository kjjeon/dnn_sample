import argparse
import sys
import tensorflow as tf


FLAGS = None


def main(_):
    print(FLAGS.input_dir)
    print(FLAGS.output_dir)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/banana/input', help='input file directory')
    parser.add_argument('--output_dir', type=str, default='/home/banana/output', help='output file directory')
    FLAGS, unparsed = parser.parse_known_args()
    print(unparsed)  # 파씽 되지 않는 args 리스트
    tf.app.run(argv=[sys.argv[0]] + unparsed)


main()

