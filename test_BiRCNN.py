# @Time : 2019/9/25 16:19
# @Author : LJW
import numpy as np
import os
import math
import shutil
import json
import pandas as pd
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, SimpleRNN, LSTM, Convolution1D, MaxPooling1D, Embedding, merge, Reshape, \
    Conv1D, Concatenate, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from sklearn.model_selection import KFold, GroupKFold
from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score,\
                                        roc_auc_score, roc_curve, auc, classification_report,confusion_matrix
from sklearn import preprocessing
import tensorflow as tf


# from keras.utils.visualize_util import plot

os.environ['CUDA_VISIBLE_DEVICES']='5'
config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True


seeds = 7
np.random.seed(seed=seeds)

from keras import backend as K
tf.set_random_seed(1234)
session = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(session)


def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''

        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        #tf.where：函数有三个参数，根据第一个条件是否成立，当为True的时候选择第二个参数中的值，否则使用第三个参数中的值。
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        # balanced_fl = tf.reduce_mean(balanced_fl,axis=1)
        balanced_fl = tf.reduce_sum(balanced_fl,axis=-1)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)
        # ce =  K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)
        return fianal_loss

    return focal_loss_fixed

def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
    :param y_true: tensor,ground truth one-hot vector shape of [batch_size, nb_class]
    :param y_pred: tensor,prediction after softmax shape of [batch_size, nb_class]
    :param alpha:
    :param gamma:
    :return:
    """
    # # parameters
    # alpha = 0.25
    # gamma = 2

    # To avoid divided by zero
    epsilon = K.epsilon()

    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    # Cross entropy
    # ce = -y_true * np.log(y_pred)
    ce = -y_true * K.log(y_pred)

    # Not necessary to multiply y_true(cause it will multiply with CE which has set unconcerned index to zero ),
    # but refer to the definition of p_t, we do it
    # weight = np.power(1 - y_pred, gamma) * y_true
    weight = K.pow(1 - y_pred, gamma) * y_true

    # Now fl has a shape of [batch_size, nb_class]
    # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
    # (CE has set unconcerned index to zero)
    #
    # alpha_step = tf.where(y_true, alpha*np.ones_like(y_true), 1-alpha*np.ones_like(y_true))
    #fl shape:(batch_size,nb_class)
    fl = ce * weight * alpha


    # Both reduce_sum and reduce_max are ok
    reduce_fl = K.sum(fl, axis=-1)


    return reduce_fl


def model_init(x_lead1_train, x_lead2_train, x_train_HRV):
    '''
    The model is demonstrated at Model.png
    The Model class used with functional API is applied in this case

    Input:
    x_lead1_train: 3D Array(None, time_step, beat_length), stand for lead I data
    x_lead2_train: 3D Array(None, time_step, beat_length), stand for lead II data
    x_train_HRV:3D Array(None, time_step, hrv_length), stand for HRV data



    Output:

    model: the main model for the proposed algorithm

    model_CNN : CNN model is used to monitor the CNN features

    model_RNN: RNN model is used to monitor the RNN features
    '''

    ECG_input_length = x_lead1_train.shape[1]
    ECG_input_dim = x_lead1_train.shape[2]
    HRV_input_length = x_train_HRV.shape[1]
    HRV_input_dim = x_train_HRV.shape[2]
    class_num = 3
    ECG_nb_features = 128
    HRV_nb_features = 128

    # nb_filter0 = 128
    # filter_length0 = 36
    # pool_length0 = 16
    nb_filter0 = 64
    filter_length0 = 16
    pool_length0 = 3

    ECG_fm = Input(shape=(ECG_input_length, ECG_input_dim), name='ECG_fm')
    ECG_fm2 = Input(shape=(ECG_input_length, ECG_input_dim), name='ECG_fm2')
    HRV_fm = Input(shape=(HRV_input_length, HRV_input_dim), name='HRV_fm')

    # CNN model begins here
    # The input tensor have to be extended into 4-D tensor at first, even the coefficient at last dimension is 1

    ECG_in = Reshape((ECG_input_length, ECG_input_dim, 1),
                     input_shape=(ECG_input_length, ECG_input_dim),name='ECG_fm/reshape')(ECG_fm)
    ECG_in2 = Reshape((ECG_input_length, ECG_input_dim, 1),
                      input_shape=(ECG_input_length, ECG_input_dim),name='ECG_fm2/reshape')(ECG_fm2)

    # In each Time Step, a 1-D CNN is used to decompose the input ECG data into local CNN features
    ECG_in = TimeDistributed(Convolution1D(filters=nb_filter0,
                                           kernel_size=filter_length0,
                                           padding='valid',
                                           activation='relu'),name='ECG_fm/conv1')(ECG_in)
    ECG_in = BatchNormalization(axis=-1, epsilon=1.001e-5, name='ECG_fm/bn1')(ECG_in)
    ECG_in = TimeDistributed(Convolution1D(filters=nb_filter0,
                                           kernel_size=filter_length0,
                                           padding='valid',
                                           activation='relu'),name='ECG_fm/conv2')(ECG_in)
    ECG_in = BatchNormalization(axis=-1, epsilon=1.001e-5, name='ECG_fm/bn2')(ECG_in)
    ECG_in = TimeDistributed(Convolution1D(filters=nb_filter0,
                                           kernel_size=filter_length0,
                                           padding='valid',
                                           activation='relu'),name='ECG_fm/conv3')(ECG_in)

    ECG_in = TimeDistributed(MaxPooling1D(pool_size=pool_length0), name='ECG_fm/max_pool')(ECG_in)
    # The output CNN features are stored here to display the extracted CNN features
    ECG_Conv = ECG_in
    ECG_in = Dropout(0.5, name='ECG_fm/dropout')(ECG_in)

    ECG_in2 = TimeDistributed(Convolution1D(filters=nb_filter0,
                                            kernel_size=filter_length0,
                                            padding='valid',
                                            activation='relu'),name='ECG_fm2/conv1')(ECG_in2)
    ECG_in2 = BatchNormalization(axis=-1, epsilon=1.001e-5, name='ECG_fm2/bn1')(ECG_in2)
    ECG_in2 = TimeDistributed(Convolution1D(filters=nb_filter0,
                                            kernel_size=filter_length0,
                                            padding='valid',
                                            activation='relu'),name='ECG_fm2/conv2')(ECG_in2)
    ECG_in2 = BatchNormalization(axis=-1, epsilon=1.001e-5, name='ECG_fm2/bn2')(ECG_in2)
    ECG_in2 = TimeDistributed(Convolution1D(filters=nb_filter0,
                                            kernel_size=filter_length0,
                                            padding='valid',
                                            activation='relu'),name='ECG_fm2/conv3')(ECG_in2)

    ECG_in2 = TimeDistributed(MaxPooling1D(pool_size=pool_length0),name='ECG_fm2/max_pool')(ECG_in2)
    ECG_in2 = Dropout(0.5,name='ECG_fm2/dropout')(ECG_in2)
    # (?, 5, 16, 128)
    print('ECG_in before flatten:', ECG_in.shape)
    # TimeDistributed层输入至少为3D，且第一个维度应该是时间所表示的维度。
    ECG_in = TimeDistributed(Flatten(), name='ECG_fm/flatten')(ECG_in)
    ECG_in2 = TimeDistributed(Flatten(), name='ECG_fm2/flatten')(ECG_in2)
    # (?, 5, 2048)
    print('ECG_in after flatten:', ECG_in.shape)

    # If you want to replace the CNN model for speeding up by DNN model, you can use the codes as follows

    # ECG_in = TimeDistributed(Dense(ECG_nb_features, activation='relu'))(ECG_fm)
    # ECG_Conv = ECG_in
    # ECG_in = Dropout(0.25)(ECG_in)
    #
    # ECG_in2 = TimeDistributed(Dense(ECG_nb_features, activation='relu'))(ECG_fm2)
    # ECG_in2 = Dropout(0.25)(ECG_in2)


    # An BiRNN model is composed by 2 RNN network with the opposite directions
    ECG_forward = SimpleRNN(ECG_nb_features, activation='relu', name='ECG_fm/rnn_forward')(ECG_in)
    ECG_backward = SimpleRNN(ECG_nb_features, activation='relu', go_backwards=True, name='ECG_fm/rnn_backward')(ECG_in)

    ECG_forward2 = SimpleRNN(ECG_nb_features, activation='relu', name='ECG_fm2/rnn_forward')(ECG_in2)
    ECG_backward2 = SimpleRNN(ECG_nb_features, activation='relu', go_backwards=True, name='ECG_fm2/rnn_backward')(ECG_in2)
    # (?, 128)
    print('ECG_forward.output_shape:',ECG_forward.shape)
    # ECG_BRCNN1 = merge([ECG_forward, ECG_backward], mode='concat')
    # ECG_BRCNN2 = merge([ECG_forward2, ECG_backward2], mode='concat')
    ECG_BRCNN1 = Concatenate(axis=-1, name='ECG_fm/concat')([ECG_forward, ECG_backward])
    ECG_BRCNN2 = Concatenate(axis=-1, name='ECG_fm2/concat')([ECG_forward2, ECG_backward2])
    # (?, 256)
    print('ECG_BRCNN1.output_shape:', ECG_BRCNN1.shape)



    # RNN features are output here
    RNN_features = ECG_BRCNN1

    # ECG_BRCNN = merge([ECG_BRCNN1, ECG_BRCNN2], mode='concat')
    ECG_BRCNN = Concatenate(axis=-1,  name='ECG_fm_ECG_fm2/concat')([ECG_BRCNN1, ECG_BRCNN2])
    ECG_BRCNN = BatchNormalization(axis=-1, epsilon=1.001e-5, name='ECG_fm_ECG_fm2/bn')(ECG_BRCNN)


    ECG_BRCNN = Dense(ECG_nb_features, activation='relu', name='ECG_fm_ECG_fm2/Dense')(ECG_BRCNN)

    ECG_BRCNN = Dropout(0.5, name='ECG_fm_ECG_fm2/dropout')(ECG_BRCNN)

    model_HRV_forward = SimpleRNN(HRV_nb_features, activation='relu', name='HRV_fm/rnn_forward')(HRV_fm)
    model_HRV_backward = SimpleRNN(HRV_nb_features, activation='relu', go_backwards=True, name='HRV_fm/rnn_backward')(HRV_fm)

    # HRV_BRCNN = merge([model_HRV_forward, model_HRV_backward], mode='concat')
    HRV_BRCNN = Concatenate(axis=-1, name='HRV_fm/concate')([model_HRV_forward, model_HRV_backward])
    HRV_BRCNN = Dropout(0.5, name='HRV_fm/dropout')(HRV_BRCNN)

    # BRCNN = merge([ECG_BRCNN, HRV_BRCNN], mode='concat')
    BRCNN = Concatenate(axis=-1, name='ECG_fm_HRV_fm/concat')([ECG_BRCNN, HRV_BRCNN])
    BRCNN = BatchNormalization(axis=-1, epsilon=1.001e-5, name='ECG_fm_HRV_fm/bn')(BRCNN)

    BRCNN = Dense(ECG_nb_features, activation='relu', name='ECG_fm_HRV_fm/Dense')(BRCNN)
    BRCNN = Dropout(0.5, name='ECG_fm_HRV_fm/dropout')(BRCNN)

    output = Dense(class_num, activation='softmax', name='main_output')(BRCNN)

    model = Model(inputs=[ECG_fm, ECG_fm2, HRV_fm], outputs=output)
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.compile(loss=focal_loss([41458,866,3708]),
    #               optimizer=optimizer,
    #               metrics=['accuracy'])

    # model_RNN = Model(inputs=ECG_fm, outputs=RNN_features)
    #
    # model_CNN = Model(inputs=ECG_fm, outputs=ECG_Conv)

    return model


class Metrics(Callback):

    # def __init__(self, filepath, val_data ,Label):
    def __init__(self, filepath, batch_size):
        self.file_path = filepath
        self.batch_size = batch_size

        # self.val_input = val_data
        # self.val_Label = Label

    def on_train_begin(self, logs={}):
        self.best_val_f1 = 0
        self.best_val_thresh = 0
        self.stop_wait = 0
        self.lr_wait = 0
        self.erarly_stop_patience = 20
        self.reduce_lr_patience = 5
        self.min_lr = 1e-6
        self.factor = 0.2
        self.val_thresh = []
        self.val_f1 = []
        self.val_recall = []
        self.val_precision = []

    def on_epoch_end(self, epoch, logs={}):
        '''调用sklearn中的f1_score函数计算'''
        # 计算label的shape为(n,1)的f1
        # val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        # val_targ = self.validation_data[1]


        # 计算label的shape为(n,2)或one-hot编码的标签的f1
        x_input = [self.validation_data[0], self.validation_data[1],  self.validation_data[2]]
        val_predict = np.argmax(np.asarray(self.model.predict(x_input, batch_size=self.batch_size)), axis=-1)
        # val_predict = np.argmax(np.asarray(self.model.predict(self.val_input)), axis=1)
        # print('val_predict length:', len(val_predict))
        # val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        # val_predict = val_predict[:,1]
        print('len val_predict:',len(val_predict))


        val_targ = np.argmax(self.validation_data[3], axis=-1)

        # val_targ = np.argmax(self.val_Label , axis=1)
        # val_targ = self.validation_data[1]
        # val_targ = val_targ[:,1]


        # _val_auc = roc_auc_score(val_targ, val_predict)
        # fpr, tpr, thresh = roc_curve(val_targ, val_predict, pos_label=1)
        # _val_auc_1 = auc(fpr, tpr)
        # diff = tpr-fpr
        # max_diff = max(diff)
        # index = np.arange(len(diff))
        # best_index = index[diff==max_diff]
        # _val_best_thresh = thresh[best_index[0]]
        # self.val_thresh.append(_val_best_thresh )


        # val_predict = (val_predict  > _val_best_thresh) * 1

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        # self.val_f1.append(_val_f1_V)
        # self.val_recall.append(_val_recall_V)
        # self.val_precision.append(_val_precision_V)
        # print(' - val_f1_V: %.4f - val_precision_V: %.4f - val_recall_V: %.4f'
        #       % (_val_f1_V, _val_precision_V, _val_recall_V))
        print('current lr:', K.get_value(self.model.optimizer.lr))
        if _val_f1 > self.best_val_f1:
            self.model.save_weights(self.file_path, overwrite=True)
            self.best_val_f1 = _val_f1
            print("best f1: {}".format(self.best_val_f1))
            self.stop_wait = 0
            self.lr_wait = 0
        else:
            self.stop_wait += 1
            print("val f1: {}, but not the best f1".format(_val_f1))
            # self.lr_wait += 1

            # if self.lr_wait >= self.reduce_lr_patience:
            #     old_lr = float(K.get_value(self.model.optimizer.lr))
            #     if old_lr > self.min_lr:
            #         new_lr = old_lr * self.factor
            #         new_lr = max(new_lr, self.min_lr)
            #         K.set_value(self.model.optimizer.lr, new_lr)
            #         self.lr_wait = 0
            #         print('====\nEpoch %05d: ReduceLROnPlateau reducing '
            #                  'learning rate to %s.=====' % (epoch + 1, new_lr))

            if self.stop_wait >= self.erarly_stop_patience:
                print("======Epoch %d: Early Stopping!!=============" % epoch)
                self.model.stop_training = True




def training_vis(hist, save_path, i=0):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    iters = [i + 1 for i in range(0, len(loss))]
    plt.figure()
    plt.subplot(211)
    plt.plot(iters, loss, 'r', label='train_loss')
    plt.plot(iters, val_loss, 'g', label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="best")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(iters, acc, 'b', label='train_acc')
    plt.plot(iters, val_acc, 'k', label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(save_path + str(i) + '.png')

def data_load(train_data_dir, train_record_list, test_data_dir, test_record_list):
    train_data_list = os.listdir(train_data_dir)
    test_data_list = os.listdir(test_data_dir)
    maxlen = 5
    beat_length = 361
    hrv_length = 2
    class_num = 3
    x_train_ECG1 = np.array([]).reshape((0, maxlen, beat_length))
    x_train_ECG2 = np.array([]).reshape((0, maxlen, beat_length))
    x_train_HRV = np.array([]).reshape((0, maxlen, hrv_length))
    y_train_c4 = np.array([]).reshape((0, class_num))

    x_val_ECG1 = np.array([]).reshape((0, maxlen, beat_length))
    x_val_ECG2 = np.array([]).reshape((0, maxlen, beat_length))
    x_val_HRV = np.array([]).reshape((0, maxlen, hrv_length))
    y_val_c4 = np.array([]).reshape((0, class_num))

    x_test_ECG1 = np.array([]).reshape((0, maxlen, beat_length))
    x_test_ECG2 = np.array([]).reshape((0, maxlen, beat_length))
    x_test_HRV = np.array([]).reshape((0, maxlen, hrv_length))
    y_test_c4 = np.array([]).reshape((0, class_num))

    for rec in train_data_list:
        if int(rec.strip('.npz')) in train_record_list[:20]:
            a = np.load(train_data_dir + rec)
            ECG1 = a['ECG1']
            ECG2 = a['ECG2']
            HRV = a['HRV']
            Label = a['Label']

            x_train_ECG1 = np.concatenate((x_train_ECG1, ECG1), axis=0)
            x_train_ECG2 = np.concatenate((x_train_ECG2, ECG2), axis=0)
            x_train_HRV = np.concatenate((x_train_HRV, HRV), axis=0)
            y_train_c4 = np.concatenate((y_train_c4, Label), axis=0)
            print('TARIN:', rec)
        else:
            a = np.load(train_data_dir + rec)
            ECG1 = a['ECG1']
            ECG2 = a['ECG2']
            HRV = a['HRV']
            Label = a['Label']

            x_val_ECG1 = np.concatenate((x_val_ECG1, ECG1), axis=0)
            x_val_ECG2 = np.concatenate((x_val_ECG2, ECG2), axis=0)
            x_val_HRV = np.concatenate((x_val_HRV, HRV), axis=0)
            y_val_c4 = np.concatenate((y_val_c4, Label), axis=0)
            print('VAL:', rec)

    for rec in test_data_list:
        if int(rec.strip('.npz')) in test_record_list:
            a = np.load(test_data_dir + rec)
            ECG1 = a['ECG1']
            ECG2 = a['ECG2']
            HRV = a['HRV']
            Label = a['Label']
            x_test_ECG1 = np.concatenate((x_test_ECG1, ECG1), axis=0)
            x_test_ECG2 = np.concatenate((x_test_ECG2, ECG2), axis=0)
            x_test_HRV = np.concatenate((x_test_HRV, HRV), axis=0)
            y_test_c4 = np.concatenate((y_test_c4, Label), axis=0)
            print('TEST:', rec)

    np.savez('Train', ECG1=x_train_ECG1, ECG2=x_train_ECG2, HRV=x_train_HRV, Label=y_train_c4)
    np.savez('Val', ECG1=x_val_ECG1, ECG2=x_val_ECG2, HRV=x_val_HRV, Label=y_val_c4)
    np.savez('Test', ECG1=x_test_ECG1, ECG2=x_test_ECG2, HRV=x_test_HRV, Label=y_test_c4)
    print('data load finished!')


def main():
    train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
                         205, 207, 208, 209, 215, 220, 223, 230]
    test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
                        219, 221, 222, 228, 231, 232, 233, 234]


    train_data_dir = '../EMBC/train_bircnn_3_channel/'
    test_data_dir = '../EMBC/test_bircnn_3_channel/'
    mode = 0
    batch_size = 64
    nb_epoch = 50
    maxlen = 5
    beat_length = 361
    hrv_length = 2
    class_num = 3
    tmp_dir = 'exp4/'
    weight_path = 'exp4/weights/'
    train_results_save_path = 'exp4/train_results/'
    train_results_img_save_path = 'exp4/train_results/imgs/'

    if not (os.path.exists(train_results_img_save_path)):
        os.makedirs(train_results_img_save_path)
    if not (os.path.exists(train_results_save_path)):
        os.makedirs(train_results_save_path)
    if not (os.path.exists(weight_path)):
        os.makedirs(weight_path)

    # data_load(train_data_dir, train_record_list, test_data_dir, test_record_list)
    Train = np.load('Train.npz')
    x_train_ECG1 = Train['ECG1']
    x_train_ECG2 = Train['ECG2']
    x_train_HRV = Train['HRV']
    y_train_c4 = Train['Label']
    Val= np.load('Val.npz')
    x_val_ECG1 = Val['ECG1']
    x_val_ECG2 = Val['ECG2']
    x_val_HRV = Val['HRV']
    y_val_c4 = Val['Label']
    Test = np.load('Test.npz')
    x_test_ECG1 = Test['ECG1']
    x_test_ECG2 = Test['ECG2']
    x_test_HRV = Test['HRV']
    y_test_c4 = Test['Label']


    print('x_train_ECG1.shape:', x_train_ECG1.shape)
    print('x_train_ECG2.shape:', x_train_ECG2.shape)
    print('x_train_HRV.shape:', x_train_HRV.shape)
    print('y_train_c4.shape:', y_train_c4.shape)
    print('x_val_ECG1.shape:', x_val_ECG1.shape)
    print('x_val_ECG2.shape:', x_val_ECG2.shape)
    print('x_val_HRV.shape:', x_val_HRV.shape)
    print('y_val_c4.shape:', y_val_c4.shape)
    print('x_test_ECG1.shape:', x_test_ECG1.shape)
    print('x_test_ECG2.shape:', x_test_ECG2.shape)
    print('x_test_HRV.shape:', x_test_HRV.shape)
    print('y_test_c4.shape:', y_test_c4.shape)

    '''训练集:共46032个样本,N:41458,S:866,V:3308,F:400
       验证集:共4849个样本,N:4288,S:73,V:474,F:14
       测试集:共49573个样本,N:44138,S:1833,V:3214,F:388
    '''
    num2char = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}
    for i in range(class_num):
        print('训练集里' + num2char[i] + '类的数量：', int(sum(y_train_c4[:, i])))
    for i in range(class_num):
        print('验证集里' + num2char[i] + '类的数量：', int(sum(y_val_c4[:, i])))
    for i in range(class_num):
        print('测试集里' + num2char[i] + '类的数量：', int(sum(y_test_c4[:, i])))

    # 对训练集、验证集进行shuffle
    index1 = np.arange(y_train_c4.shape[0])
    np.random.shuffle(index1)
    x_train_ECG1 = x_train_ECG1[index1]
    x_train_ECG2 = x_train_ECG2[index1]
    x_train_HRV= x_train_HRV[index1]
    y_train_c4 = y_train_c4[index1]

    index2 = np.arange(y_val_c4.shape[0])
    np.random.shuffle(index2)
    x_val_ECG1 = x_val_ECG1[index2]
    x_val_ECG2 = x_val_ECG2[index2]
    x_val_HRV = x_val_HRV[index2]
    y_val_c4 = y_val_c4[index2]



    model = model_init(x_train_ECG1, x_train_ECG2, x_train_HRV)
    # plot_model(model, to_file='BiRCNN.png',show_shapes=True)
    # model.save(tmp_dir + 'model.h5')
    # model.save(tmp_dir + 'model_CNN.h5')
    # model.save(tmp_dir + 'model_RNN.h5')
    model.summary()
    #



    metrics_1 = Metrics(weight_path+'best_f1_model.h5', batch_size=batch_size)
    History = model.fit([x_train_ECG1, x_train_ECG2, x_train_HRV], y_train_c4,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        callbacks=[metrics_1],
                        validation_data=([x_val_ECG1, x_val_ECG2, x_val_HRV], y_val_c4),
                        )

    # F1 = metrics_1.best_val_f1
    # shutil.copyfile('models/best_f1_model.h5', os.path.join(weight_path, 'fold_' + str(i)
    #                                                        + '_best_f1_' + str(F1) + '_model.h5'))
    # training_vis(History, train_results_img_save_path, 0)
    # i = i + 1

    model = model_init(x_test_ECG1, x_test_ECG2, x_test_HRV)
    model.load_weights(os.path.join(weight_path, 'best_f1_model.h5'))

    # f1_result = test_model(model, x_test_ECG1, x_test_ECG2, x_test_HRV, y_test_c4, batch_size)


    pred_test_label = model.predict([x_test_ECG1, x_test_ECG2, x_test_HRV], batch_size=batch_size)
    y_pred = np.argmax(pred_test_label, axis=-1)
    y_true = np.argmax(y_test_c4, axis=-1)
    print(np.unique(y_pred))
    print(y_true[:10])
    print(y_pred[:10])
    print(classification_report(y_true, y_pred, target_names=['N', 'S', 'V']))
    print('========== onfusion matrix ==========')
    print(confusion_matrix(y_true, y_pred))








if __name__ == "__main__":
    main()