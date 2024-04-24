import tensorflow as tf 
import numpy as np

import conv_tf as cvf
import hhar_plot

import time
import math
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

layers = tf.keras.layers 

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6#len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 64
TOTAL_ITER_NUM = 1000000000

select = 'a'

metaDict = {'a':[119080, 1193], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))

###### Import training data
def read_audio_csv(value):
    line = value
    defaultVal = [[0.] for idx in range(WIDE * FEATURE_DIM + OUT_DIM)]
    fileData = tf.io.decode_csv(line, record_defaults=defaultVal)
    features = fileData[:WIDE * FEATURE_DIM]
    features = tf.reshape(features, [WIDE, FEATURE_DIM])
    labels = fileData[WIDE * FEATURE_DIM:]
    return features, labels


def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
    # Create a dataset from filenames
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Shuffle if needed
    if shuffle_sample:
        dataset = dataset.shuffle(buffer_size=len(filenames))

    # Read and parse each CSV line
    dataset = dataset.flat_map(tf.data.TextLineDataset)
    dataset = dataset.map(read_audio_csv)

    # Repeat for multiple epochs
    if num_epochs is not None:
        dataset = dataset.repeat(num_epochs)

    # Batch the data
    dataset = dataset.batch(batch_size)

    # Prefetch for performance (optional)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def batch_norm_layer(inputs, train, scope=None):
	
    inputs = tf.expand_dims(inputs, axis=-1)

    batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
    if train:
        outputs = batch_norm(inputs, training=True)
    else:
        outputs = batch_norm(inputs, training=False)

    # Remove the dummy channel dimension
    outputs = tf.squeeze(outputs, axis=-1)

    return outputs
	
class DropoutGRUCell(tf.keras.layers.GRUCell):
	def __init__(self, units, dropout_rate, **kwargs):
		super().__init__(units, **kwargs)
		self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

	def call(self, inputs, states, training=None):
		outputs, new_states = super().call(inputs, states, training=training)
		outputs = self.dropout_layer(outputs, training=training)
		return outputs, new_states
	
gru_cell1 = DropoutGRUCell(INTER_DIM, dropout_rate=0.5)
gru_cell2 = DropoutGRUCell(INTER_DIM, dropout_rate=0.5)

def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.name_scope(name):
		used = tf.sign(tf.reduce_max(tf.abs(inputs), axis=2))  # (BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, axis=1)  # (BATCH_SIZE)
		
		mask = tf.sign(tf.reduce_max(tf.abs(inputs), axis=2, keepdims=True))
		mask = tf.tile(mask, [1, 1, INTER_DIM])  # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, axis=1)  # (BATCH_SIZE, INTER_DIM)
		# inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
		sensor_inputs = tf.expand_dims(inputs, axis=3)
		# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
		acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)

		acc_conv1 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, 2*3*CONV_LEN),
                                   strides=(1, 2*3), padding='VALID', activation=None, data_format='channels_last')(acc_inputs)
		acc_conv1 = batch_norm_layer(acc_conv1, train, scope='acc_BN1')
		acc_conv1 = tf.nn.relu(acc_conv1)
		acc_conv1_shape = acc_conv1.shape
		acc_conv1 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(acc_conv1, training=train)

		acc_conv2 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, CONV_LEN_INTE),
										strides=(1, 1), padding='VALID', activation=None, data_format='channels_last')(acc_conv1)
		acc_conv2 = batch_norm_layer(acc_conv2, train, scope='acc_BN2')
		acc_conv2 = tf.nn.relu(acc_conv2)
		acc_conv2_shape = acc_conv2.shape
		acc_conv2 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(acc_conv2, training=train)

		acc_conv3 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, CONV_LEN_LAST),
										strides=(1, 1), padding='VALID', activation=None, data_format='channels_last')(acc_conv2)
		acc_conv3 = batch_norm_layer(acc_conv3, train, scope='acc_BN3')
		acc_conv3 = tf.nn.relu(acc_conv3)
		acc_conv3_shape = acc_conv3.shape
		acc_conv_out = tf.reshape(acc_conv3, [acc_conv3_shape[0], acc_conv3_shape[1], 1, acc_conv3_shape[2], acc_conv3_shape[3]])

		gyro_conv1 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, 2*3*CONV_LEN),
                                    strides=(1, 2*3), padding='VALID', activation=None, data_format='channels_last')(gyro_inputs)
		gyro_conv1 = batch_norm_layer(gyro_conv1, train, scope='gyro_BN1')
		gyro_conv1 = tf.nn.relu(gyro_conv1)
		gyro_conv1_shape = gyro_conv1.shape
		gyro_conv1 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(gyro_conv1, training=train)

		gyro_conv2 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, CONV_LEN_INTE),
											strides=(1, 1), padding='VALID', activation=None, data_format='channels_last')(gyro_conv1)
		gyro_conv2 = batch_norm_layer(gyro_conv2, train, scope='gyro_BN2')
		gyro_conv2 = tf.nn.relu(gyro_conv2)
		gyro_conv2_shape = gyro_conv2.shape
		gyro_conv2 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(gyro_conv2, training=train)

		gyro_conv3 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, CONV_LEN_LAST),
											strides=(1, 1), padding='VALID', activation=None, data_format='channels_last')(gyro_conv2)
		gyro_conv3 = batch_norm_layer(gyro_conv3, train, scope='gyro_BN3')
		gyro_conv3 = tf.nn.relu(gyro_conv3)
		gyro_conv3_shape = gyro_conv3.shape
		gyro_conv_out = tf.reshape(gyro_conv3, [gyro_conv3_shape[0], gyro_conv3_shape[1], 1, gyro_conv3_shape[2], gyro_conv3_shape[3]])

		sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
		senor_conv_shape = sensor_conv_in.get_shape().as_list()	
		sensor_conv_in_shape = sensor_conv_in.shape
		sensor_conv_in = tf.reshape(sensor_conv_in, [sensor_conv_in_shape[0], sensor_conv_in_shape[1], sensor_conv_in_shape[2] * sensor_conv_in_shape[3], sensor_conv_in_shape[4]])
		
		sensor_conv1 = tf.keras.layers.Conv2D(CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN),
                                      strides=(1, 1), padding='SAME', activation=None, data_format='channels_last')(sensor_conv_in)
		
		sensor_conv1 = batch_norm_layer(sensor_conv1, train, scope='sensor_BN1')
		sensor_conv1 = tf.nn.relu(sensor_conv1)
		sensor_conv1_shape = sensor_conv1.shape
		sensor_conv1 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(sensor_conv1, training=train)

		sensor_conv2 = tf.keras.layers.Conv2D(CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN2),
											strides=(1, 1), padding='SAME', activation=None, data_format='channels_last')(sensor_conv1)
		sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope='sensor_BN2')
		sensor_conv2 = tf.nn.relu(sensor_conv2)
		sensor_conv2_shape = sensor_conv2.shape
		sensor_conv2 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(sensor_conv2, training=train)

		sensor_conv3 = tf.keras.layers.Conv2D(CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN3),
											strides=(1, 1), padding='SAME', activation=None, data_format='channels_last')(sensor_conv2)
		sensor_conv3 = batch_norm_layer(sensor_conv3, train, scope='sensor_BN3')
		sensor_conv3 = tf.nn.relu(sensor_conv3)
		sensor_conv3_shape = sensor_conv3.shape
		sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2] * sensor_conv3_shape[3]])


		cell = tf.keras.layers.StackedRNNCells([gru_cell1, gru_cell2])

		cell_output, final_stateTuple, _ = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)(sensor_conv_out, mask=mask)


		sum_cell_out = tf.reduce_sum(cell_output * mask, axis=1, keepdims=False)
		avg_cell_out = sum_cell_out/avgNum

		logits = tf.keras.layers.Dense(OUT_DIM, activation=None, name='output')(avg_cell_out)

		return logits

tfrecord_train_files = ["/Users/rosshhun/Downloads/DeepSense/sepHARData_a/train.tfrecord"]
tfrecord_eval_files = ["/Users/rosshhun/Downloads/DeepSense/sepHARData_a/eval.tfrecord"]

global_step = tf.Variable(0, trainable=False)

dataset = cvf.input_pipeline_har(tfrecord_train_files, BATCH_SIZE)
eval_dataset = cvf.input_pipeline_har(tfrecord_eval_files, BATCH_SIZE, shuffle_sample=False)

# Create iterators for the training and evaluation datasets
dataset_iter = iter(dataset)
eval_dataset_iter = iter(eval_dataset)

# Initialize iteration counter
iteration = 0

# Training loop
while True:
    try:
        batch_feature, batch_label = next(dataset_iter)
        logits = deepSense(batch_feature, True, name='deepSense')

        predict = tf.argmax(logits, axis=1)

        batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
        loss = tf.reduce_mean(batchLoss)

        # Get trainable variables
        t_vars = []
        for layer in [gru_cell1, gru_cell2]:
            t_vars += layer.trainable_variables

        # Add regularization
        regularizers = 0.
        for var in t_vars:
            regularizers += tf.nn.l2_loss(var)
        loss += 5e-4 * regularizers

        # Calculate gradients using GradientTape
        with tf.GradientTape() as tape:
            logits = deepSense(batch_feature, True, name='deepSense')
            batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
            loss = tf.reduce_mean(batchLoss) + 5e-4 * regularizers

        gradients = tape.gradient(loss, t_vars)

        discOptimizer = tf.keras.optimizers.Adam()
        discOptimizer.apply_gradients(zip(gradients, t_vars))

        lossV = loss.numpy()
        _trainY = batch_label.numpy()
        _predict = np.argmax(logits.numpy(), axis=1)
        _label = np.argmax(_trainY, axis=1)
        _accuracy = np.mean(_label == _predict)
        hhar_plot.plot('train cross entropy', lossV)
        hhar_plot.plot('train accuracy', _accuracy)

        if iteration % 100 == 0:
            dev_accuracy = []
            dev_cross_entropy = []

    except tf.errors.OutOfRangeError:
        break

    except StopIteration:
        print("Reached end of dataset.")
        break

# Evaluation loop
for eval_idx in range(EVAL_ITER_NUM):
    try:
        batch_eval_feature, batch_eval_label = next(eval_dataset_iter)
        logits_eval = deepSense(batch_eval_feature, False, reuse=True, name='deepSense')
        predict_eval = tf.argmax(logits_eval, axis=1)
        loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))

        eval_loss_v = loss_eval.numpy()
        _trainY = batch_eval_label.numpy()
        _predict = predict_eval.numpy()
        _label = np.argmax(_trainY, axis=1)
        _accuracy = np.mean(_label == _predict)
        dev_accuracy.append(_accuracy)
        dev_cross_entropy.append(eval_loss_v)

        hhar_plot.plot('train cross entropy', loss_eval.numpy())
        hhar_plot.plot('train accuracy', _accuracy)

        if iteration % 10 == 0:
            for acc, ce in zip(dev_accuracy, dev_cross_entropy):
                hhar_plot.plot('dev accuracy', acc)
                hhar_plot.plot('dev cross entropy', ce)

        iteration += 1

        hhar_plot.tick()
        
    except tf.errors.OutOfRangeError:
        break
    except StopIteration:
        print("Reached end of dataset.")
        break