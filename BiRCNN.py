import numpy as np
import os
import math
import shutil
import json
import pandas as pd
import pickle
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, SimpleRNN, LSTM, Convolution1D, MaxPooling1D, Embedding, merge, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from sklearn.model_selection import KFold, GroupKFold
from imblearn.over_sampling import SMOTE, ADASYN
from keras import backend as K
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score,\
                                        roc_auc_score, roc_curve, auc, classification_report
from sklearn import preprocessing

from ..my_DenseNet import DenseNet_model_init
# from keras.utils.visualize_util import plot

os.environ['CUDA_VISIBLE_DEVICES']='0'



# 1: add TP_S,FN_S,FP_S,FNS
# 2: change validation rate : 0.1 -> 0.2 -> 0.3
# 3: CNN 1 layer -> 3 layers,  nb_epoch: 100 -> 50

def test_model(models, x_test1, x_test2, x_test_HRV, y_test, batch_size):
    '''
    This function is used to test performances of the proposed algorithm

    Input parameter:
        models:     NNs model that defined at function model_init()
        x_test1:    Test Lead I ECG DATA
        x_test2:    Test Lead II ECG DATA
        x_test_HRV: Test HRV DATA
        y_test:     Label sequence
        batch_size: Defined at main function

    Output:

        Classification results of V class and S class are printed at Command Window:
        which are including Acc,Se,Sp and Fpp

    '''
    input_num = x_test1.shape[0]

    # scores = models.evaluate(
    #     [x_test1, x_test2, x_test_HRV], y_test, batch_size=batch_size)
    #
    # predict_probabilities = models.predict(
    #     [x_test1, x_test2, x_test_HRV], batch_size=batch_size)
    scores = models.evaluate(
        x_test1, y_test, batch_size=batch_size)

    predict_probabilities = models.predict(
        x_test1, batch_size=batch_size)
    accuracies = scores[1]

    print('使用sklearn计算的结果：\n')
    y_pred_1 = np.zeros((input_num, 4))
    for i in range(0, input_num):
        if np.argmax(predict_probabilities[i]) == 0:
            y_pred_1[i,0] = 1
        elif np.argmax(predict_probabilities[i]) == 1:
            y_pred_1[i,1] = 1
        elif np.argmax(predict_probabilities[i]) == 2:
            y_pred_1[i,2] = 1
        else:
            y_pred_1[i,3] = 1

    f1 = f1_score(y_test, y_pred_1,average=None)
    print('使用多标签计算方法得到的f1:')
    f1_results = {'N_f1':f1[3],'S_f1':f1[2],'V_f1':f1[1],'F_V1':f1[0]}
    print(f1_results)
    print('使用分类报告得到的结果:')
    y_pred = [np.argmax(y) for y in predict_probabilities]
    y_true = [np.argmax(y) for y in y_test]
    target_names = ['F','V','S','N']
    print(classification_report(y_true, y_pred, target_names=target_names))


    print('原始的评分结果:' + '\n')

    TP_N = 0
    TP_S = 0
    TP_V = 0
    TP_F = 0
    FP_N = 0
    FP_S = 0
    FP_V = 0
    FP_F = 0
    FN_N = 0
    FN_S = 0
    FN_V = 0
    FN_F = 0
    TN_S = 0
    TN_N = 0
    TN_V = 0
    TN_F = 0

    TP_S_ind = []
    FN_S_ind = []
    FNS = [0]*4
    FP_S_ind = []

    for mm in range(0, input_num):
        predict_classes = np.argmax(predict_probabilities[mm, :])

        if y_test[mm][3] == 1:

            if predict_classes == 3:

                TP_N += 1
                TN_S += 1
                TN_V += 1
                TN_F += 1

            elif predict_classes == 2:

                FN_N += 1
                FP_S += 1
                FP_S_ind.append(mm)

            elif predict_classes == 1:

                FN_N += 1
                FP_V += 1

            elif predict_classes == 0:

                FN_N += 1
                FP_F += 1

        elif y_test[mm][2] == 1:

            if predict_classes == 3:

                FN_S += 1
                FN_S_ind.append(mm)
                FNS[3] = FNS[3] + 1
                FP_N += 1

            elif predict_classes == 2:

                TP_S += 1
                TP_S_ind.append(mm)
                TN_N += 1
                TN_V += 1
                TN_F += 1

            elif predict_classes == 1:

                FN_S += 1
                FN_S_ind.append(mm)
                FNS[1] = FNS[1] + 1
                FP_V += 1

            elif predict_classes == 0:

                FN_S += 1
                FN_S_ind.append(mm)
                FNS[0] = FNS[0] + 1
                FP_F += 1

        elif y_test[mm][1] == 1:

            if predict_classes == 3:

                FN_V += 1
                FP_N += 1

            elif predict_classes == 2:

                FN_V += 1
                FP_S += 1
                FP_S_ind.append(mm)

            elif predict_classes == 1:

                TP_V += 1
                TN_S += 1
                TN_N += 1
                TN_F += 1

            elif predict_classes == 0:

                FN_V += 1
                FP_F += 1

        elif y_test[mm][0] == 1:

            if predict_classes == 3:

                FN_F += 1
                FP_N += 1

            elif predict_classes == 2:

                FN_F += 1
                FP_S += 1
                FP_S_ind.append(mm)

            elif predict_classes == 1:

                FN_F += 1
                FP_V += 1

            elif predict_classes == 0:

                TP_F += 1
                TN_S += 1
                TN_V += 1
                TN_N += 1

    '''
     The '+1' at denominator is for preventing the denominator to be 0
    '''
    VEB_Acc = float(TP_V + TN_V) / (TP_V + TN_V + FP_V + FN_V + 1)

    VEB_SE = float(TP_V) / (TP_V + FN_V + 1)

    VEB_SP = float(TN_V) / (TN_V + FP_V + 1)

    VEB_PP = float(TP_V) / (TP_V + FP_V + 1)

    VEB_F1 = 2*VEB_SE*VEB_PP / (VEB_SE+VEB_PP)

    SVEB_Acc = float(TP_S + TN_S) / (TP_S + TN_S + FP_S + FN_S + 1)

    SVEB_SE = float(TP_S) / (TP_S + FN_S + 1)

    SVEB_SP = float(TN_S) / (TN_S + FP_S + 1)

    SVEB_PP = float(TP_S) / (TP_S + FP_S + 1)

    SVEB_F1 = 2 * SVEB_SE * SVEB_PP / (SVEB_SE + SVEB_PP)

    # if accuracies > 0.9:
    print('N,S,V,F: ', TP_N, TP_S, TP_V, TP_F, TN_N, TN_S, TN_V,
          TN_F, FN_N, FN_S, FN_V, FN_F, FP_N, FP_S, FP_V, FP_F)
    print('\n\nVEB_Acc:', VEB_Acc)
    print('VEB_SE:', VEB_SE)
    print('VEB_SP:', VEB_SP)
    print('VEB_PP:', VEB_PP)
    print('VEB_F1:', VEB_F1)

    print('\nSVEB_Acc:', SVEB_Acc)
    print('SVEB_SE:', SVEB_SE)
    print('SVEB_SP:', SVEB_SP)
    print('SVEB_PP:', SVEB_PP)
    print('SVEB_F1:', SVEB_F1)
    print('\n')

    # return accuracies, predict_probabilities, predict_classes, TP_S_ind, FN_S_ind, FP_S_ind, FNS
    return f1_results


def data_load(path, tt_path):
    '''
    This function is used to input the ECG and HRV data

    x or y means Data or Label
    train or test means the Usage of data

    ECG1: Lead I ECG data
    ECG2: Lead II ECG data

         ECG data are stored with shape (Number of data, Time steps, Dimension of each beat)
         which is (66347, 5, 361) in training stage, and (33182, 5, 361) in testing stage

    HRV: HRV data
         It's shape is defined as (Number of data, Time steps, Two adjacent HRVs of current ECG beat)
         (66347, 11,2) in tranning stage and (33182, 11,2) in test stage

    y_train_4,y_test_4: Labels with 4 class, N S V F

    '''

    #x_train_ECG1 = np.transpose(np.load(path + 'x_train_ECG1.npy'))
    #x_test_ECG1 = np.transpose(np.load(path + 'x_test_ECG1.npy'))
    #x_train_ECG2 = np.transpose(np.load(path + 'x_train_ECG2.npy'))
    #x_test_ECG2 = np.transpose(np.load(path + 'x_test_ECG2.npy'))
    #x_train_HRV = np.transpose(np.load(path + "x_train_HRV.npy"))
    #x_test_HRV = np.transpose(np.load(path + "x_test_HRV.npy"))
    #y_test_4 = np.transpose(np.load(path + "y_test_c4.npy"))
    #y_train_4 = np.transpose(np.load(path + "y_train_c4.npy"))

    x_train_ECG1 = np.load(path + 'x_train_ECG1.npy')
    x_test_ECG1 = np.load(tt_path + 'x_test_ECG1.npy')
    x_train_ECG2 = np.load(path + 'x_train_ECG2.npy')
    x_test_ECG2 = np.load(tt_path + 'x_test_ECG2.npy')
    x_train_HRV = np.load(path + "x_train_HRV.npy")
    x_test_HRV = np.load(tt_path + "x_test_HRV.npy")
    y_test_4 = np.load(tt_path + "y_test_c4.npy")
    y_train_4 = np.load(path + "y_train_c4.npy")

    return x_train_ECG1, x_test_ECG1, x_train_HRV, x_test_HRV, y_train_4, y_test_4, x_train_ECG2, x_test_ECG2


def model_init(x_lead1_train, x_lead2_train, x_train_HRV):
    '''
    The model is demonstrated at Model.png

    The Model class used with functional API is applied in this case

    Output:

    model: the main model for the proposed algorithm

    model_CNN : CNN model is used to monitor the CNN features

    model_RNN: RNN model is used to monitor the RNN features
    '''

    ECG_input_length = x_lead1_train.shape[1]
    ECG_input_dim = x_lead1_train.shape[2]
    HRV_input_length = x_train_HRV.shape[1]
    HRV_input_dim = x_train_HRV.shape[2]

    ECG_nb_features = 128
    HRV_nb_features = 128

    nb_filter0 = 128
    filter_length0 = 36
    pool_length0 = 16

    ECG_fm = Input(shape=(ECG_input_length, ECG_input_dim), name='ECG_fm')
    ECG_fm2 = Input(shape=(ECG_input_length, ECG_input_dim), name='ECG_fm2')
    HRV_fm = Input(shape=(HRV_input_length, HRV_input_dim), name='HRV_fm')

    # CNN model begins here
    # The input tensor have to be extended into 4-D tensor at first, even the coefficient at last dimension is 1

    ECG_in = Reshape((ECG_input_length, ECG_input_dim, 1),
                     input_shape=(ECG_input_length, ECG_input_dim))(ECG_fm)
    ECG_in2 = Reshape((ECG_input_length, ECG_input_dim, 1),
                      input_shape=(ECG_input_length, ECG_input_dim))(ECG_fm2)

    # In each Time Step, a 1-D CNN is used to decompose the input ECG data into local CNN features
    ECG_in = TimeDistributed(Convolution1D(nb_filter=nb_filter0,
                                           filter_length=filter_length0,
                                           border_mode='valid',
                                           activation='relu'))(ECG_in)
    ECG_in = TimeDistributed(Convolution1D(nb_filter=nb_filter0,
                                           filter_length=filter_length0,
                                           border_mode='valid',
                                           activation='relu'))(ECG_in)
    ECG_in = TimeDistributed(Convolution1D(nb_filter=nb_filter0,
                                           filter_length=filter_length0,
                                           border_mode='valid',
                                           activation='relu'))(ECG_in)
    ECG_in = TimeDistributed(MaxPooling1D(pool_length=pool_length0))(ECG_in)
    # The output CNN features are stored here to display the extracted CNN features
    ECG_Conv = ECG_in
    ECG_in = Dropout(0.5)(ECG_in)

    ECG_in2 = TimeDistributed(Convolution1D(nb_filter=nb_filter0,
                                            filter_length=filter_length0,
                                            border_mode='valid',
                                            activation='relu'))(ECG_in2)
    ECG_in2 = TimeDistributed(Convolution1D(nb_filter=nb_filter0,
                                            filter_length=filter_length0,
                                            border_mode='valid',
                                            activation='relu'))(ECG_in2)
    ECG_in2 = TimeDistributed(Convolution1D(nb_filter=nb_filter0,
                                            filter_length=filter_length0,
                                            border_mode='valid',
                                            activation='relu'))(ECG_in2)
    ECG_in2 = TimeDistributed(MaxPooling1D(pool_length=pool_length0))(ECG_in2)
    ECG_in2 = Dropout(0.5)(ECG_in2)

    ECG_in = TimeDistributed(Flatten())(ECG_in)
    ECG_in2 = TimeDistributed(Flatten())(ECG_in2)

    # If you want to replace the CNN model for speeding up by DNN model, you can use the codes as follows
    '''
    ECG_in = TimeDistributed(Dense(ECG_nb_features, activation='relu'))(ECG_fm)
    ECG_Conv = ECG_in
    ECG_in = Dropout(0.25)(ECG_in)

    ECG_in2 = TimeDistributed(Dense(ECG_nb_features, activation='relu'))(ECG_fm2)
    ECG_in2 = Dropout(0.25)(ECG_in2)
    '''

    # An BiRNN model is composed by 2 RNN network with the opposite directions
    ECG_forward = SimpleRNN(ECG_nb_features, activation='relu')(ECG_in)
    ECG_backward = SimpleRNN(
        ECG_nb_features, activation='relu', go_backwards=True)(ECG_in)

    ECG_forward2 = SimpleRNN(ECG_nb_features, activation='relu')(ECG_in2)
    ECG_backward2 = SimpleRNN(
        ECG_nb_features, activation='relu', go_backwards=True)(ECG_in2)

    ECG_BRCNN1 = merge([ECG_forward, ECG_backward], mode='concat')
    ECG_BRCNN2 = merge([ECG_forward2, ECG_backward2], mode='concat')

    # RNN features are output here
    RNN_features = ECG_BRCNN1

    ECG_BRCNN = merge([ECG_BRCNN1, ECG_BRCNN2], mode='concat')

    ECG_BRCNN = Dense(ECG_nb_features, activation='relu')(ECG_BRCNN)

    ECG_BRCNN = Dropout(0.25)(ECG_BRCNN)

    model_HRV_forward = SimpleRNN(HRV_nb_features, activation='relu')(HRV_fm)
    model_HRV_backward = SimpleRNN(
        HRV_nb_features, activation='relu', go_backwards=True)(HRV_fm)

    HRV_BRCNN = merge([model_HRV_forward, model_HRV_backward], mode='concat')
    HRV_BRCNN = Dropout(0.25)(HRV_BRCNN)

    BRCNN = merge([ECG_BRCNN, HRV_BRCNN], mode='concat')
    BRCNN = Dense(ECG_nb_features, activation='relu')(BRCNN)
    BRCNN = Dropout(0.25)(BRCNN)

    output = Dense(4, activation='softmax', name='main_output')(BRCNN)

    model = Model(input=[ECG_fm, ECG_fm2, HRV_fm],
                  output=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model_RNN = Model(input=ECG_fm, output=RNN_features)

    model_CNN = Model(input=ECG_fm, output=ECG_Conv)

    return model, model_CNN, model_RNN


class Metrics(Callback):

    # def __init__(self, filepath, val_data ,Label):
    def __init__(self, filepath):
        self.file_path = filepath
        # self.val_input = val_data
        # self.val_Label = Label

    def on_train_begin(self, logs={}):
        self.best_val_f1 = 0
        self.best_val_thresh = 0
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
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
        # val_predict = np.argmax(np.asarray(self.model.predict(self.val_input)), axis=1)
        # print('val_predict length:', len(val_predict))
        # val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        # val_predict = val_predict[:,1]


        val_targ = np.argmax(self.validation_data[1], axis=1)
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
        # _val_f1_V = f1_score(val_targ, val_predict, average=None)[1]
        # _val_recall_V = recall_score(val_targ, val_predict, average=None)[1]
        # _val_precision_V = precision_score(val_targ, val_predict, average=None)[1]
        _val_f1_S = f1_score(val_targ, val_predict, average=None)[2]
        _val_recall_S= recall_score(val_targ, val_predict, average=None)[2]
        _val_precision_S = precision_score(val_targ, val_predict, average=None)[2]
        # self.val_f1.append(_val_f1_V)
        # self.val_recall.append(_val_recall_V)
        # self.val_precision.append(_val_precision_V)
        # print(' - val_f1_V: %.4f - val_precision_V: %.4f - val_recall_V: %.4f'
        #       % (_val_f1_V, _val_precision_V, _val_recall_V))

        print(' - val_f1_S: %.4f - val_precision_S: %.4f - val_recall_S: %.4f'
              % (_val_f1_S, _val_precision_S, _val_recall_S))

        # print(' - best_thresh: %.4f - val_auc_1: %.4f - val_f1: %.4f - val_precision: %.4f - val_recall: %.4f'
        #       % (_val_best_thresh, _val_auc_1, _val_f1, _val_precision, _val_recall))
        # _val_f1 = (_val_f1_V + _val_f1_S )/2
        _val_f1 = _val_f1_S
        if _val_f1 > self.best_val_f1:
            self.model.save_weights(self.file_path, overwrite=True)
            self.best_val_f1 = _val_f1
            print("best f1: {}".format(self.best_val_f1))
        else:
            print("val f1: {}, but not the best f1".format(_val_f1))
        return

def training_vis(hist, save_path, i):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    result1 = dict()
    result1['loss'] = loss
    result1['val_loss'] = val_loss
    result1['acc'] = acc
    result1['val_acc'] = val_acc

    with open(os.path.join(save_path, 'train_result_' + str(i) +'.pickle'), 'wb') as fin:
        pickle.dump(result1, fin)


def main():

    mode = 0

    batch_size = 256
    nb_epoch = 50
    tmp_dir = 'exp4/'
    weight_path = 'exp4/weights/'
    train_results_save_path = 'exp4/train_results/'
    train_results_img_save_path = 'exp4/train_results/imgs/'
    if not (os.path.exists(train_results_img_save_path)):
        os.makedirs(train_results_img_save_path)

    if not (os.path.exists(train_results_save_path)):
        os.makedirs(train_results_save_path)
    if not (os.path.exists(weight_path)):  # if the weight path doesn't exist, create weights
        os.makedirs(weight_path)

    # weight_files = os.listdir(weight_path)

    # data_dir = 'record44_N8000_2tr1te_4/'
    #save_path = 'result/'
    # data_dir = 'EMBC/exp0/record44_N8000_2tr1te_0/'
    Train_Data = np.load('EMBC/Train/DS1.npz')
    x_train_ECG1 = Train_Data['ECG1']
    x_train_ECG2 = Train_Data['ECG2']
    x_train_HRV = Train_Data['HRV']
    y_train_4 = Train_Data['Label_c4']
    '''(51229, 5, 361)
       (51229, 11, 2)
       (51229, 4)样本标签为中间位置心跳的标签'''
    # 对训练集进行shuffle
    index = np.arange(y_train_4.shape[0])
    random.shuffle(index)
    x_train_ECG1 = x_train_ECG1[index]
    x_train_ECG2 = x_train_ECG2[index]
    x_train_HRV = x_train_HRV[index]
    y_train_4 = y_train_4[index]
    # 归一化为0均值,单位方差
    # scaler1 = preprocessing.StandardScaler().fit(x_train_ECG1)
    # x_train_ECG1 = scaler1.transform(x_train_ECG1)
    # scaler2 = preprocessing.StandardScaler().fit(x_train_ECG2)
    # x_train_ECG2 = scaler2.transform(x_train_ECG2)
    # scaler3 = preprocessing.StandardScaler().fit(x_train_HRV)
    # x_train_HRV = scaler3.transform(x_train_HRV)

    f = open('train_group.json', 'r')
    train_group = json.load(f)
    groups = []
    for i in range(0, len(train_group)):
        for j in range(0, train_group[i]):
            groups.append(i)
    # print(groups)
    print('groups的长度：',len(groups))

    ##测试集
    Test_Data = np.load('EMBC/Test/DST1.npz')
    x_test_ECG1 = Test_Data['ECG1']
    x_test_ECG2 = Test_Data['ECG2']
    x_test_HRV = Test_Data['HRV']
    y_test_4 = Test_Data['Label_c4']

    # x_test_ECG1 = scaler1.transform(x_test_ECG1)
    # x_test_ECG2 = scaler2.transform(x_test_ECG2)
    # x_test_HRV = scaler3.transform(x_test_HRV)


    # path = '/home/xiepw/EMBC/syn/'+tmp_dir+data_dir
    # path = data_dir
    # tt_path = data_dir
    # tt_path = '/home/xiepw/EMBC/syn/'+tmp_dir+data_dir+'test2/trte/'
    # (x_train_ECG1, x_test_ECG1, x_train_HRV, x_test_HRV, y_train_4,
    #  y_test_4, x_train_ECG2, x_test_ECG2) = data_load(path, tt_path)

    # batch_size = 512

    gkf = GroupKFold(n_splits=2)
    test_f1_results = {}
    i = 0
    if mode == 0:
        print('##########Test##########')
        for train, val in gkf.split(x_train_ECG1, y_train_4, groups=groups):

            # print(train)
            # print(val)
            train_ECG1 = x_train_ECG1[train]
            train_ECG2 = x_train_ECG2[train]
            train_HRV = x_train_HRV[train]
            train_Label = y_train_4[train]
            val_ECG1 = x_train_ECG1[val]
            val_ECG2 = x_train_ECG2[val]
            val_HRV = x_train_HRV[val]
            val_Label = y_train_4[val]


            # model, model_CNN, model_RNN = model_init(x_train_ECG1, x_train_ECG2, x_train_HRV)
            # model, model_CNN, model_RNN = model_init(train_ECG1, train_ECG2, train_HRV)
            model = DenseNet_model_init()
            # model.save(tmp_dir + 'model.h5')
            # model.save(tmp_dir + 'model_CNN.h5')
            # model.save(tmp_dir + 'model_RNN.h5')

            # val_input = [val_ECG1, val_ECG2, val_HRV]
            # metrics_1 = Metrics('models/best_f1_model.h5', val_input, val_Label )
            metrics_1 = Metrics('models/best_f1_model.h5')


            # History = model.fit([x_train_ECG1, x_train_ECG2, x_train_HRV], y_train_4, batch_size=batch_size,
            #                     nb_epoch=nb_epoch, callbacks=[checkpointers],
            #                     validation_split=0.1, shuffle=True, class_weight=[1,1,1,1]) #{0:1., 1:30., 2:13., 3:90.}

            # History = model.fit([train_ECG1, train_ECG2, train_HRV], train_Label, batch_size=batch_size,
            History = model.fit(train_ECG1, train_Label, batch_size=batch_size,
                                epochs=nb_epoch,
                                callbacks=[metrics_1, EarlyStopping(monitor='val_loss', patience=4,mode='auto')],
                                # validation_data=([val_ECG1, val_ECG2, val_HRV], val_Label),
                                validation_data=(val_ECG1, val_Label),
                                shuffle=True, class_weight='auto')

            F1 = metrics_1.best_val_f1
            shutil.copyfile('models/best_f1_model.h5', os.path.join(weight_path, 'fold_' + str(i)
                                                                   + '_best_f1_' + str(F1) + '_model.h5'))
            training_vis(History, train_results_save_path, i)
            i = i + 1



    elif mode == 1:

        print('##########Test##########')
        # model = load_model(tmp_dir + 'model.h5')
        model = DenseNet_model_init()
        weight_list = os.listdir(weight_path)
        for filename in weight_list:
            # if(int(filename[7:9])>20):
            #     print(filename)
                model.load_weights(os.path.join(weight_path,filename))
                # load weightsh + 'predict_class', {'predict_class': predict_class}) # predicted_class (0,1,2,3,4)redicted_class (0,1,2,3,4)path + 'predict_class', {'predict_class': predict_class})  # predicted_class (0,1,2,3,4)
                # accuracy, predict_probability, predict_class, TP_S_ind, FN_S_ind, FP_S_ind, FNS = \
                #     test_model(model, x_test_ECG1, x_test_ECG2, x_test_HRV, y_test_4, batch_size)
                f1_result = test_model(model, x_test_ECG1, x_test_ECG2, x_test_HRV, y_test_4, batch_size)
                test_f1_results[filename] = f1_result

        print(test_f1_results)

    else:
        print('######绘制训练过程中的loss和acc曲线######')
        train_imgs = os.listdir(train_results_img_save_path)
        if len(train_imgs) != 0:
            for img in train_imgs:
                os.remove(os.path.join(train_results_img_save_path, img))

        train_reuslts = os.listdir(train_results_save_path)
        i = 0
        for res in train_reuslts:
            if res.endswith('.pickle'):
                f = open(train_results_save_path + res, 'rb')
                data = pickle.load(f)
                loss = data['loss']
                acc = data['acc']
                val_loss = data['val_loss']
                val_acc = data['val_acc']
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
                plt.savefig(train_results_img_save_path + str(i) + '.png')
                i = i + 1



        # x_demo1 is the selected test data with shape (1,5,361)
        # x_demo = x_train_ECG1[0, :, :]
        # x_demo1 = x_demo.reshape(1, 5, 361)

        # CNN_output is a 3-D matrix with shape (5,15,128)
        # which means (number of beats, dimension of each ECG feature,  feature number)
        # you can use surfc(double(CNN_output(:,:,k))) to review the k-th CNN feature by matlab
        # CNN_output = model_CNN.predict(x_demo1)
        # sio.savemat(tmp_dir + 'x_demo1.mat', {'x_demo1': x_demo1[0]})
        # sio.savemat(tmp_dir + 'CNN_output.mat', {'CNN_output': CNN_output[0]})




if __name__ == "__main__":
    main()