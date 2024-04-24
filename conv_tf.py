import os

import tensorflow as tf
import numpy as np

# todo: change the path to your own data folder path
TF_RECORD_PATH = r'/Users/rosshhun/Downloads/DeepSense/sepHARData_a'
DATA_FOLDER_PATH = r'/Users/rosshhun/Downloads/DeepSense/sepHARData_a'

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
WIDE = 20
OUT_DIM = 6#len(idDict)
BATCH_SIZE = 64


def csv_to_example(fname):
    text = np.loadtxt(fname, delimiter=',')
    features = text[:WIDE*FEATURE_DIM]
    label = text[WIDE*FEATURE_DIM:]
    # Ensure features are flattened. 
    # tf.reshape(features, [WIDE, FEATURE_DIM]) is not needed if we are flattening it right away.
    features_flat = features.flatten()  # This ensures features is a flat list suitable for FloatList.

    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
        # Use the flattened features here
        'example': tf.train.Feature(float_list=tf.train.FloatList(value=features_flat.tolist()))
    }))

    return example

def read_and_decode(tfrec_path):
    dataset = tf.data.TFRecordDataset([tfrec_path])
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, features={
        'label': tf.io.FixedLenFeature([OUT_DIM], tf.float32),
        'example': tf.io.FixedLenFeature([WIDE*FEATURE_DIM], tf.float32),
    }))
    return dataset

def input_pipeline_har(tfrec_path, batch_size, shuffle_sample=True):
    dataset = read_and_decode(tfrec_path)

    if shuffle_sample:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)

    def preprocess_batch(batch):
        example_batch = batch['example']
        label_batch = batch['label']

        # Get the batch size from the shape of example_batch
        batch_size = tf.shape(example_batch)[0]

        # Reshape example_batch to the desired shape
        example_batch = tf.reshape(example_batch, (batch_size, WIDE, FEATURE_DIM))

        return example_batch, label_batch

    dataset = dataset.map(preprocess_batch)

    return dataset

def main():
    writer = tf.io.TFRecordWriter(os.path.join(TF_RECORD_PATH, 'train.tfrecord'))
    train_path = os.path.join(DATA_FOLDER_PATH, 'train')
    train_files = os.listdir(train_path)
    for f in train_files:
        f_pre, f_suf = f.split('.')
        if f_suf == 'csv':
            f_path = os.path.join(train_path, f)
            example = csv_to_example(f_path)
            writer.write(example.SerializeToString())
    writer.close()

    writer = tf.io.TFRecordWriter(os.path.join(TF_RECORD_PATH, 'eval.tfrecord'))
    eval_path = os.path.join(DATA_FOLDER_PATH, 'eval')
    eval_files = os.listdir(eval_path)
    for f in eval_files:
        f_pre, f_suf = f.split('.')
        if f_suf == 'csv':
            f_path = os.path.join(eval_path, f)
            example = csv_to_example(f_path)
            writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()