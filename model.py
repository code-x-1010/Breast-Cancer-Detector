import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import load_img

dic = dict()
dic['lateriality'] = 'L'
dic['view'] = 'CC'
dic['age'] = 63
dic['biospy'] = 1
dic['invasive'] = 0
dic['implant'] = 0
dic['density'] = 'B'
dic['path'] = './rsna-breast-cancer-512-pngs/8785_1050149859.png'


def process_data(dic):
    density = {'B':2,'A':1,'C':3,'D':4,np.nan:0}
    dic['density'] = density[dic['density']]
    lat = {'L':0,'R':1}
    view = {'CC':0,'MLO':1,'ML':2,'AT':3}
    dic['lateriality'] = lat[dic['lateriality']]
    dic['view'] = view[dic['view']]
    model = tf.keras.models.load_model('./mammography_pred_model.h5')
    img = load_img(dic['path'])
    arr = np.asarray(img)
    arr = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2]) / 255
    dic['path'] = model.predict(arr)[0][0]
    arr =[]
    for  i in dic.values():
        arr.append(float(i))
    return np.array(arr).reshape((1,-1))
classifier = pickle.load(open('decision_model.sav', 'rb'))
temp = process_data(dic)
# print(temp)
pred = classifier.predict(temp)
print('cancer presence = ',pred[0])