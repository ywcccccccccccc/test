import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

L = 128
load_path = 'pred_confusion_mat_L' + str(L) + '.npy'
data = np.load(load_path)

EsNoLow = 0
EsNoHigh = 18
Gap = 3
nClass = data.shape[1]

num_point = (int)((EsNoHigh-EsNoLow)/Gap) + 1
snr = np.zeros(num_point)
acc = np.zeros(num_point)
cnt_sum = 0
for rows in data[0, :]:
    for ele in rows:
        cnt_sum = cnt_sum + ele
for i in range(num_point):
    snr[i] = EsNoLow + i*Gap
    cnt_acc = 0
    for j in range(nClass):
        cnt_acc = cnt_acc + data[i, j, j]
    acc[i] = cnt_acc/cnt_sum

plt.plot(snr, acc, 'o-', color='blue', label='L = ' + str(L))

my_x_ticks = np.arange(EsNoLow,EsNoHigh+3,3)
plt.xticks(my_x_ticks)
plt.ylim((0.8, 1.01))
plt.xlabel('SNR(dB)', fontsize=15)
plt.ylabel('Probability of Detection', fontsize=15)

plt.savefig('L' + str(L) + '.png', format='png', dpi=300, bbox_inches = 'tight')
res = np.vstack((snr, acc))
np.savetxt('acc.txt', res, fmt='%.04f')
plt.show()
