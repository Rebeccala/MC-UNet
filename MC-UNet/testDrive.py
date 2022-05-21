import os
import cv2
import imageio
import numpy as np
from sklearn.metrics import  recall_score, roc_auc_score, accuracy_score, confusion_matrix
from keras.callbacks import  ModelCheckpoint

# from scipy.misc.pilutil import *
import math
from util import *
data_location = ''

testing_images_loc =data_location + '(DAC+SPP)/DRIVE/test/images/'
testing_label_loc =data_location + '(DAC+SPP)/DRIVE/test/labels/'

# testing_images_loc =data_location + '(DAC+SPP)/DRIVE/preprocess/WF(20)/test/image/'
# testing_label_loc =data_location + '(DAC+SPP)/DRIVE/preprocess/WF(20)/test/label/'

test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []

test_data_add = []
desired_size=592
for i in test_files:
    im = cv2.imread(testing_images_loc + i)
    im = np.array(im)

    # label = imageio.imread(testing_label_loc + i.split('_')[0] + '_test.gif')
    label = imageio.imread(testing_label_loc + i.split('_')[0] + '_manual1.png',pilmode="L")

    # print(testing_images_loc + i)
    # print(testing_label_loc + i.split('_')[0] + '_manual1.png')

    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    test_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    # Change '_manual1.tiff' to the label name
    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)

test_data = np.array(test_data)
test_label = np.array(test_label)


x_test = test_data.astype('float32') / 255.
y_test = test_label.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

y_test = crop_to_shape(y_test,(len(y_test), 584, 565, 1))

from  SA_UNet_SPP import *
from  SA_UNet_DAC import *
from  SA_UNet_ce import *
from  SA_UNet import *

model=SA_UNet_kmp(input_size=(desired_size,desired_size,3),start_neurons=16,lr=1e-3,keep_prob=1,block_size=1)
model.summary()
# weight="Model/testmodel/Ablation/DRIVE/①lr 1e-3 epoch 100 lr 1e-4 epoch 50/SA_UNet_SPP.h5"
# weight="Model/testmodel/Ablation/DRIVE/②lr 1e-3 epoch 100 lr 1e-4 epoch 100/SA_UNet_DAC.h5"
# weight="Model/testmodel/Ablation/DRIVE/③lr 1e-3 epoch 150 lr 1e-4 epoch 50/SA_UNet_ce.h5"
# weight="Model/DRIVE/SA_UNet_pre.h5"
# weight="Model/DRIVE/testmodel/Ablation/DRIVE/①lr 1e-3 epoch 100 lr 1e-4 epoch 50/SA_UNet_DAC.h5"
# weight="Model/DRIVE/SA_UNet_DAC_pre.h5"
# weight="Model/DRIVE/SA_UNet_SPP.h5"
# weight="Model/DRIVE/SA_UNet_SPP_pre.h5"
# weight="Model/DRIVE/SA_UNet_ce.h5"
# weight="Model/DRIVE/SA_UNet_ce_pre.h5"

weight="Model/DRIVE/SA_UNet_3kmp.h5"

model.load_weights(weight)
model_checkpoint = ModelCheckpoint(weight, monitor='val_acc', verbose=1, save_best_only=True)

y_pred = model.predict(x_test)
y_pred= crop_to_shape(y_pred,(20,584,565,1))
y_pred_threshold = []
y_pred_threshold1 = []
i=0
yyy=[]
for y in y_pred:

    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    y_pred_threshold1.append(temp)
    # yy = np.ravel(y)
    # print(np.ravel(yy))
    yyy.append(y)

    y = y * 255
    # cv2.imwrite('F:/PycharmProjects/code/testROCimage/1Unet/DRIVE/probilitity/%d.png' % (i + 1), y)
    # cv2.imwrite('(DAC+SPP)/DRIVE/testresult/3/pre/%d.png' % (i + 1), y)
    i+=1
y_test = list(np.ravel(y_test))
y_pred_threshold = list(np.ravel(yyy))
y_pred_threshold1 = list(np.ravel(y_pred_threshold1))
# print(y_pred_threshold)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold1).ravel()
from sklearn import metrics
import matplotlib.pylab as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_threshold, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
print(roc_auc)

plt.plot(fpr, tpr, 'B', label='DRIVE AUC = %0.4f' % roc_auc)
plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')  # 横坐标是fpr
plt.ylabel('True Positive Rate')  # 纵坐标是tpr
plt.title('Receiver operating characteristic')
plt.show()

print('Accuracy:', accuracy_score(y_test, y_pred_threshold1))
print('Sensitivity:', recall_score(y_test, y_pred_threshold1))
print('Specificity:', tn / (tn + fp))
print('AUC:', roc_auc_score(y_test, list(np.ravel(y_pred))))
print("F1:",2*tp/(2*tp+fn+fp))
N=tn+tp+fn+fp
S=(tp+fn)/N
P=(tp+fp)/N
print("MCC:",(tp/N-S*P)/math.sqrt(P*S*(1-S)*(1-P)))