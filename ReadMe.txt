rain.npz文件: 包含训练集的20个record(除去223、230),每个record的每个导联上的前后2个心跳组成一个样本，前后2个RR间隔组成一个样本,
              'ECG1'_shape:(46032,5,361),'ECG2'_shape:(46032,5,361),'HRV'_shape:(46032,5,2),'Label'_shape:(46032,4)

Val.npz文件: 包含训练集中的223、230两个record,每个record的每个导联上的前后2个心跳组成一个样本，前后2个RR间隔组成一个样本,
              'ECG1'_shape:(4849,5,361),'ECG2'_shape:(4849,5,361),'HRV'_shape:(4849,5,2),'Label'_shape:(4849,4)

Test.npz文件: 包含测试集的22个record,每个record的每个导联上的前后2个心跳组成一个样本，前后2个RR间隔组成一个样本,
              'ECG1'_shape:(49573,5,361),'ECG2'_shape:(49573,5,361),'HRV'_shape:(49573,5,2),'Label'_shape:(49573,4)

用sigmoid函数提取心跳，对每个心跳进行z-score归一化，对每个5*2的HRV('2'代表当前的RR间隙及下一个RR间隙，
'5'代表前后两个心跳)除以该2D矩阵的最大值进行缩放;