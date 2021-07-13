
# @Time : 2019/9/24 14:56
# @Author : LJW

import wfdb
import numpy as np
import scipy.io as sio
import os
import json
import pickle
import random
import dtcwt
import math
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from scipy.signal import medfilt
np.set_printoptions(suppress = True)
import warnings
import traceback
import time
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, SimpleRNN, LSTM, Convolution1D, MaxPooling1D, Embedding, Merge, Reshape,Dense, Dropout, \
        BatchNormalization, LeakyReLU, concatenate,GRU,merge, Conv2D,MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential,save_model, load_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D,Conv2D, MaxPooling2D,\
    AveragePooling2D, GlobalAveragePooling2D
import pickle
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils import plot_model
from keras.layers import Layer
from keras.regularizers import l2
from keras import initializers
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score
from decimal import Decimal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

tf.set_random_seed(1234)
os.environ['CUDA_VISIBLE_DEVICES']='6'

def sigmf(x, a, c):
    return 1/(1 + np.exp(-a*(x-c)))

def wtmedian_denoise(sig, gain_mask = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], baseline0_windows = 36, baseline1_windows = 108):
    transform = dtcwt.Transform1d()
    #sig是列向量
    sig_t = transform.forward(sig, nlevels=len(gain_mask))

    sig_recon = transform.inverse(sig_t, gain_mask)
    #200ms和600ms的中值滤波（采样率位360Hz)
    baseline0 = medfilt(sig_recon, baseline0_windows*2+1)
    baseline1 = medfilt(baseline0, baseline1_windows*2+1)
    sig_denoised = sig_recon - baseline1
    # baseline0 = [np.median(sig_recon[max(0, x - 36):min(x + 36, len(sig_recon) - 1)])
    #              for x in range(len(sig_recon))]
    # baseline1 = [np.median(baseline0[max(
    #     0, x - 108):min(x + 108, len(baseline0) - 1)]) for x in range(len(baseline0))]
    # sig_denoised = list(map((lambda x, y: x - y), sig_recon, baseline1))
    return sig_denoised

def generate_EMBC_mitdb_new(data_dir, data_list, save_dir):
    # data_dir = '../MIT-BIH_arrythimia/'
    # train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
    #                      205, 207, 208, 209, 215, 220, 223, 230]
    # test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
    #                     219, 221, 222, 228, 231, 232, 233, 234]
    '''生成类似于EMBC/mitdb里的txt文件'''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num = 0
    for cur_record_id in data_list:
        f = open(save_dir + str(cur_record_id) + '.txt','w')
        print(cur_record_id)

        record_ann = wfdb.rdann(os.path.join(data_dir, str(cur_record_id)), 'atr')
        samples = record_ann.annsamp
        symbols = record_ann.anntype

        for i in range(1, len(symbols)):
            # r_position_pre, label_pre, rr_interval, beat_label, r_position
            r_position_pre = samples[i-1]
            label_pre = symbols[i-1]

            rr_interval = Decimal((samples[i] - samples[i-1])/360).quantize(Decimal('0.000'))
            beat_label = symbols[i]
            r_position = samples[i]
            if i == 1:
                print(symbols[i-1])
            line = str(r_position_pre) + '  ' + label_pre + '   ' + str(rr_interval) + '    ' + beat_label + \
                   '    ' + str(r_position)
            f.write(line + '\n')
        num = num + 1

        print('=====Processing第' + str(num) + '个record=====')


def get_bircnn_inputs_data(data_list, save_path):
    # train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
    #                      205, 207, 208, 209, 215, 220, 223, 230]
    # test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
    #                     219, 221, 222, 228, 231, 232, 233, 234]
    data_path = '../EMBC/MITDB/'
    fs = 360
    delete_p = 180
    TimeStep_half_ECG = 2
    TimeStep_half_HRV = 2
    num_class = 3
    anno_path = '../EMBC/mitdb_new/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # anno_list = os.listdir(anno_path)
    # for anno_rec in anno_list:
    # AAMI2
    label_t4 = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
                'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
                'V': 'V', 'E': 'V', 'F': 'V',
                # 'F': 'F',
                # 'P': 'Q', 'f':'Q', 'U':'Q'
                }

    ECG1 = np.array([]).reshape((0, 2 * TimeStep_half_ECG + 1, fs + 1))
    ECG2 = np.array([]).reshape((0, 2 * TimeStep_half_ECG + 1, fs + 1))
    HRV = np.array([]).reshape((0, 2 * TimeStep_half_ECG + 1, 2))
    Label_c4 = np.array([]).reshape((0, 4))
    num = 0
    for rec in data_list:
        num = num + 1
        print('=====Processing第' + str(num) + '个record=====')

        print(rec)
        # r_position_pre = []
        # label_pre = []
        # rr_interval = []
        beat_label = []
        r_position = []
        # f = open(anno_path + str(rec) + '.txt', 'r')
        # lines = f.readlines()
        # for line in lines:
        #     a = line.split()
        #     r_position_pre.append(int(a[0]))
        #     label_pre.append(a[1])
        #     rr_interval.append(a[2])
        #     beat_label.append(a[3])
        #     r_position.append(int(a[4]))
        record_ann = wfdb.rdann(os.path.join('../../MIT-BIH_arrythimia/', str(rec)), 'atr')
        samples = record_ann.annsamp
        symbols = record_ann.anntype
        for i, symbol in enumerate(symbols):
            if symbol in label_t4.keys():
                beat_label.append(symbol)
                r_position.append(samples[i])
        print('len beat_label:', len(beat_label))
        print(beat_label[0],r_position[0])
        data_raw = sio.loadmat(data_path + str(rec) + 'm.mat')['val']
        data_raw = data_raw.T
        # print(data_raw.shape)
        sig_length = data_raw.shape[0]
        ECG_w_sig = np.zeros((sig_length, 2))
        for i in range(2):
            ECG_w_sig[:,i] = wtmedian_denoise(data_raw[:, i], gain_mask = [0, 0, 1, 1, 1, 1, 1, 1, 0])
            # ECG_w_sig[:, i] = ECG_w_sig[:,i]/np.abs(np.max(ECG_w_sig[:,i]))

        xHRV = np.diff(np.array(r_position)).reshape(len(r_position)-1, 1)/360
        # xHRV = xHRV/np.max(xHRV,axis=0)
        xHRV = (xHRV - np.mean(xHRV))/ np.std(xHRV)

        # minmax = MinMaxScaler(feature_range=(0, 1))

        # xHRV = minmax.fit_transform(xHRV.T).T
        HRV_feature = np.concatenate((xHRV[:-1,0:1], xHRV[1:,0:1]), axis=1)

        # ECG_Label = beat_label[1:-1]
        ECG_Label = beat_label
        Ltmp = len(r_position) - 1

        Matrix_lead1 = np.zeros((Ltmp-1, 2 * fs+1))
        Matrix_lead2 = np.zeros((Ltmp-1, 2 * fs+1))
        ECG_feature_lead1 = np.zeros((Ltmp - 2 * TimeStep_half_ECG - 1, 2 * TimeStep_half_ECG + 1, fs + 1))
        ECG_feature_lead2 = np.zeros((Ltmp - 2 * TimeStep_half_ECG - 1, 2 * TimeStep_half_ECG + 1, fs + 1))
        Sign_Win = np.zeros((Ltmp-1, 2 * fs + 1))
        ECG_Label_4 = np.zeros((Ltmp-1, num_class))
        # ECG_feature_HRV = np.zeros(((Ltmp - 2 * TimeStep_half_HRV - 1 , 2 * TimeStep_half_HRV+ 1, 2)))
        ECG_feature_HRV = np.zeros(((Ltmp - 2 * TimeStep_half_ECG - 1, 2 * TimeStep_half_HRV + 1, 2)))

        for kk in range(1, Ltmp):
            #去掉第一跳、最后一跳
            # if ECG_Label[kk - 1] in label_t4.keys():
                # nsvf_label_index.append(kk-1)

                if kk == 1:
                    # 截r点前360点到r点波形
                    ECG_tmp1 = ECG_w_sig[max(1, r_position[kk] - fs):r_position[kk], 0]
                    L1 = len(ECG_tmp1)
                    Matrix_lead1[kk - 1, fs - L1 +1:fs + 1] = ECG_tmp1

                    Matrix_lead1[kk - 1, fs + 1:2 * fs + 1] = ECG_w_sig[r_position[kk]:r_position[kk] + fs, 1]

                    ECG_tmp1 = ECG_w_sig[max(1, r_position[kk] - fs):r_position[kk], 1]
                    Matrix_lead2[kk - 1, fs - L1 + 1:fs + 1 ] = ECG_tmp1
                    Matrix_lead2[kk - 1, fs + 1:2 * fs + 1] = ECG_w_sig[r_position[kk]:r_position[kk] + fs , 1]

                elif kk == Ltmp-1:

                    ECG_tmp1 = ECG_w_sig[r_position[kk]:min(len(ECG_w_sig), r_position[kk] + fs), 0]
                    L1 = len(ECG_tmp1)
                    Matrix_lead1[kk - 1, 0:fs + 1] = ECG_w_sig[r_position[kk] - fs:r_position[kk] + 1, 1]
                    Matrix_lead1[kk - 1, fs + 1:fs + L1 + 1] = ECG_tmp1

                    ECG_tmp1 = ECG_w_sig[r_position[kk]:min(len(ECG_w_sig), r_position[kk] + fs), 1]
                    Matrix_lead2[kk - 1, 0:fs + 1] = ECG_w_sig[r_position[kk] - fs:r_position[kk] + 1, 1]
                    Matrix_lead2[kk - 1, fs + 1:fs + L1 + 1 ] = ECG_tmp1
                else:
                    Matrix_lead1[kk - 1, 0:fs + 1] = ECG_w_sig[r_position[kk] - fs:r_position[kk] + 1, 0]
                    Matrix_lead1[kk - 1, fs + 1:2 * fs + 1 ] = ECG_w_sig[r_position[kk]:r_position[kk] + fs, 0]
                    Matrix_lead2[kk - 1, 0:fs + 1] = ECG_w_sig[r_position[kk] - fs:r_position[kk] + 1, 1]
                    Matrix_lead2[kk - 1, fs + 1:2 * fs + 1] = ECG_w_sig[r_position[kk]:r_position[kk] + fs , 1]



                L_Center = fs - 300 / 1000 * fs
                # sigmf(x, a, c])： f(x, a, c) = 1 / (1 + exp(-a(x - c)))
                Sign_L = np.array(sigmf(np.arange(fs+1), 0.1, L_Center))

                # min(0.6fs, 0.6RR(kk))
                R_Center = min(600 / 1000 * fs, 0.6 * xHRV[kk])
                Sign_R = np.ones((1, fs)) - sigmf(np.arange(fs), 0.1, R_Center)

                Sign_Win[kk - 1,:] = list(Sign_L) + list(Sign_R[0,:])

                Matrix_lead1[kk - 1,:] = Matrix_lead1[kk - 1,:]*Sign_Win[kk - 1,:]
                # Matrix_lead1[kk - 1, :] = (Matrix_lead1[kk - 1, :] - np.min(Matrix_lead1[kk - 1, :])) / (np.max(
                #     Matrix_lead1[kk - 1, :])-(np.min(Matrix_lead1[kk - 1, :])))

                Matrix_lead1[kk - 1, :] =  (Matrix_lead1[kk - 1,:] - np.mean( Matrix_lead1[kk - 1,:]))/np.std(Matrix_lead1[kk - 1,:])

                Matrix_lead2[kk - 1, :] = Matrix_lead2[kk - 1, :] * Sign_Win[kk - 1, :]
                # Matrix_lead2[kk - 1, :] = (Matrix_lead2[kk - 1, :] - np.min(Matrix_lead2[kk - 1, :])) / (np.max(
                #     Matrix_lead2[kk - 1, :]) - (np.min(Matrix_lead2[kk - 1, :])))
                Matrix_lead2[kk - 1, :] = (Matrix_lead2[kk - 1, :] - np.mean(Matrix_lead2[kk - 1, :])) / np.std(Matrix_lead2[kk - 1, :])

                beat_label_t4 = label_t4[ECG_Label[kk]]

                # if beat_label_t4 == 'N':
                #     ECG_Label_4[kk-1,:] = np.array([1, 0, 0, 0])
                # elif beat_label_t4 == 'S':
                #     ECG_Label_4[kk-1,:] = np.array([0, 1, 0, 0])
                # elif beat_label_t4 == 'V':
                #     ECG_Label_4[kk-1,:] = np.array([0, 0, 1, 0])
                # elif beat_label_t4 == 'F':
                #     ECG_Label_4[kk-1,:] = np.array([0, 0, 0, 1])
                if beat_label_t4 == 'N':
                    ECG_Label_4[kk - 1, :] = np.array([1, 0, 0])
                elif beat_label_t4 == 'S':
                    ECG_Label_4[kk - 1, :] = np.array([0, 1, 0])
                elif beat_label_t4 == 'V':
                    ECG_Label_4[kk - 1, :] = np.array([0, 0, 1])


                # ECG_feature = cell(1, 5)

                # print('Matrix_lead1.shape:', Matrix_lead1.shape)

        # nsvf_label_index = []
        for kk in range(TimeStep_half_ECG, Ltmp - TimeStep_half_ECG-1):

                # minmax = MinMaxScaler(feature_range=(0, 1))

                ECG_feature_lead1[kk - TimeStep_half_ECG, :, :] = Matrix_lead1[kk - TimeStep_half_ECG: kk + TimeStep_half_ECG + 1,
                                                              delete_p:2*fs+1 - delete_p]
                # ECG_feature_lead1[kk - TimeStep_half_HRV, :, :] = ECG_feature_lead1[kk - TimeStep_half_HRV, :, :]/\
                #                     np.max(np.max(ECG_feature_lead1[kk - TimeStep_half_HRV, :, :],axis=1),axis=0)
                # ECG_feature_lead1[kk - TimeStep_half_HRV, :, :] = minmax.fit_transform(ECG_feature_lead1[kk - TimeStep_half_HRV,:,:])

                # ECG_feature_lead1(kk - TimeStep_half_HRV,:,:) = ECG_feature_lead1(kk - TimeStep_half_HRV,:,:) / ...
                # (max(max(ECG_feature_lead1(kk - TimeStep_half_HRV,:,:)))-min(
                #     min(ECG_feature_lead1(kk - TimeStep_half_HRV,:,:)))); % normalization
                #删除最前的180点、最后的180个点，因为它们的值都接近于0
                ECG_feature_lead2[kk - TimeStep_half_ECG, :, :] = Matrix_lead2[kk - TimeStep_half_ECG: kk + TimeStep_half_ECG + 1,
                                                                delete_p:2*fs+1 - delete_p]

                # ECG_feature_lead2[kk - TimeStep_half_HRV, :, :] = ECG_feature_lead2[kk - TimeStep_half_HRV, :, :]/\
                #                     np.max(np.max(ECG_feature_lead2[kk - TimeStep_half_HRV, :, :],axis=1), axis=0)

                # ECG_feature_lead2[kk - TimeStep_half_HRV, :, :] = minmax.fit_transform(ECG_feature_lead2[kk - TimeStep_half_HRV, :, :])
                # ECG_feature_lead2(kk - TimeStep_half_HRV,:,:) = ECG_feature_lead2(kk - TimeStep_half_HRV,:,:) / ...
                # (max(max(ECG_feature_lead2(kk - TimeStep_half_HRV,:,:)))-min(
                #     min(ECG_feature_lead2(kk - TimeStep_half_HRV,:,:)))); % normalization


                HRV_f_white = HRV_feature[kk - TimeStep_half_HRV: kk + TimeStep_half_HRV + 1, :]

                # standard = StandardScaler().fit(HRV_f_white)
                # HRV_f_white = standard.transform(HRV_f_white)


                ECG_feature_HRV[kk - TimeStep_half_HRV, :, :] = HRV_f_white
                #
                # print(HRV_f_white[0])
                # print(ECG_feature_lead2[kk - TimeStep_half_HRV, :, :][0])
                # print(np.max(ECG_feature_lead2[kk - TimeStep_half_HRV, :, :], axis=1))

        # for kk in range(TimeStep_half_HRV, Ltmp - TimeStep_half_HRV-1):
        #
        #     ECG_feature_HRV[kk - TimeStep_half_HRV, :, :] = HRV_feature[kk - TimeStep_half_HRV: kk + TimeStep_half_HRV + 1, :]



        ECG_Label_C4 = ECG_Label_4[TimeStep_half_ECG :Ltmp - TimeStep_half_ECG - 1, :]

        # ECG1 = np.concatenate((ECG1, ECG_feature_lead1), axis=0)
        # ECG2 = np.concatenate((ECG2, ECG_feature_lead2), axis=0)
        # HRV = np.concatenate((HRV, ECG_feature_HRV), axis=0)
        # Label_c4 = np.concatenate((Label_c4, ECG_Label_C4), axis=0)
        print('ECG_feature_lead1.shape:', ECG_feature_lead1.shape)
        print('ECG_feature_HRV.shape:',ECG_feature_HRV.shape)
        print(ECG_Label_C4.shape)
        print(ECG_Label_C4 [:10])

        np.savez(save_path + str(rec), ECG1=ECG_feature_lead1, ECG2=ECG_feature_lead2, HRV=ECG_feature_HRV,
                 Label=ECG_Label_C4)

    # print(ECG1.shape)
    # print(ECG2.shape)
    # print(HRV.shape)
    # print(Label_c4.shape)
        # break
        # plt.subplot(411)
        # plt.plot(data_raw[:, 0][0:1000])
        # plt.grid(True)
        # plt.subplot(412)
        # plt.plot(ECG_w_sig[:,0][0:1000])
        # plt.grid(True)
        # plt.subplot(413)
        # plt.plot(data_raw[:, 1][0:1000])
        # plt.grid(True)
        # plt.subplot(414)
        # plt.plot(ECG_w_sig[:, 1][0:1000])
        # plt.grid(True)
        # plt.show()


def main():
    train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
                         205, 207, 208, 209, 215, 220, 223, 230]
    test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
                        219, 221, 222, 228, 231, 232, 233, 234]
    train_save_path = '../EMBC/train_bircnn_3_channel/'
    test_save_path = '../EMBC/test_bircnn_3_channel/'
    get_bircnn_inputs_data(train_record_list, train_save_path)
    get_bircnn_inputs_data(test_record_list, test_save_path)


    # test_label = np.array([]).reshape((0,4))
    # test_data = np.array([]).reshape((0, 5,361))
    #
    #
    # for rec in test_record_list:
    #
    #     # if int(rec.strip('.npz')) in train_record_list[:20]:
    #         a = np.load(test_save_path + str(rec)+'.npz')
    #         data = a['ECG1']
    #         label = a['Label']
    #         # print(label.shape)
    #         # data = data[:(len(data) // maxlen) * maxlen, :]
    #         # label = label[:(len(data) // maxlen) * maxlen, :]
    #         # data = [data[i:i + maxlen] for i in range(0, len(data), maxlen)]
    #         # label = [label[i:i + maxlen] for i in range(0, len(label), maxlen)]
    #         # data = np.array(data)
    #         # label = np.array(label)
    #         test_data = np.concatenate((test_data, data), axis=0)
    #         test_label = np.concatenate((test_label, label), axis=0)
    #         # print('TARIN:', rec)
    #
    #
    # print('test_label.shape:', test_label.shape)
    # print('test_data.shape:', test_data.shape)
    # num2char = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}
    # for i in range(4):
    #
    #     print('测试集里' + num2char[i] + '类的数量：', int(sum(test_label[:, i])))
    #



if __name__ == "__main__":
    main()








