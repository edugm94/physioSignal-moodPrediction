#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.02.08
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from utils.PhysioModel import PhysioModel
from utils.DatasetBuilder import DatasetBuilder
from utils.PreprocTemporalData import PreprocTemporalData

import tensorflow as tf
import time


path_to_hdf5 = 'out/2_raw_data/mood/individual/p4_raw_mood.h5'
leave_one_out = False
one_hot = True
bs = 16

ds = DatasetBuilder(
        path_to_hdf5=path_to_hdf5,
        leave_one_out=leave_one_out,
        one_hot=one_hot
    )


# Get train and test     arrays
trainData, trainLabel, testData, testLabel = ds.buildDataset()

preprocObj = PreprocTemporalData(trainData, testData)
trainData, testData = preprocObj(trainData, testData)

#Build TF Dataset
trainDataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabel))
testDataset = tf.data.Dataset.from_tensor_slices((testData, testLabel))

trainDataset = trainDataset.shuffle(buffer_size=128).batch(bs)
testDataset = testDataset.batch(bs)

# Create a model
model = PhysioModel(num_classes=3,
                    batch_size=bs)

# Create loss and metrics objects
loss_fn = tf.keras.losses.CategoricalCrossentropy()

train_acc_metric = tf.metrics.CategoricalAccuracy('train_accuracy')
test_acc_metric = tf.metrics.CategoricalAccuracy('test_accuracy')

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

log_dir = 'logs/nn_acc'
train_writer = tf.summary.create_file_writer(log_dir + '/train/')
test_writer = tf.summary.create_file_writer(log_dir + '/test/')

model_dir = 'model/nn_acc/'

@tf.function
def train_step(data, labels, step):
    with tf.GradientTape() as tape:
        pred = model(data, training=True)
        loss = loss_fn(pred, labels)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_loss(loss)
    train_acc_metric(labels, pred)

@tf.function
def test_step(data, labels, step):
    pred_test = model(data, training=False)
    loss = loss_fn(pred_test, labels)

    test_loss(loss)
    test_acc_metric(labels, pred_test)

TRACED = False
EPOCHS = 50
for epoch in range(EPOCHS):
    t = time.time()

    for step, (data, labels) in enumerate(trainDataset):
        if epoch == 0 and not TRACED:
            tf.summary.trace_on(graph=True, profiler=False)
        train_step(data, labels, epoch)
        if epoch == 0 and not TRACED:
            with train_writer.as_default():
                tf.summary.trace_export(name='graph', profiler_outdir='logs/train/', step=epoch)
            TRACED = True

    with train_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc_metric.result(), step=epoch)

    for step, (data, labels) in enumerate(testDataset):
        test_step(data, labels, epoch)
    with test_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_acc_metric.result(), step=epoch)

    template = 'ETA: {} - epoch: {}\tloss: {}\tacc: {}\tloss_test: {}\tacc_test: {}\n'
    print(template.format(
        round((time.time() - t) / 60, 2), epoch + 1, train_loss.result(), float(train_acc_metric.result()),
        test_loss.result(), float(test_acc_metric.result())
    ))

    train_loss.reset_states()
    test_loss.reset_states()
    train_acc_metric.reset_states()
    test_acc_metric.reset_states()

model.save(model_dir)