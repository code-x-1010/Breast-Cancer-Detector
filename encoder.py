import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import load_img

model = tf.keras.models.load_model('./mammography_pred_model.h5')

def prediction(path):
    global model
    img = load_img(path)
    arr = np.asarray(img)
    arr = arr.reshape(1,arr.shape[0],arr.shape[1],arr.shape[2])/255
    # print(arr)
    pred = model.predict(arr)
    print(pred)
    return pred



df = pd.read_csv('training.csv',index_col=0)

df['path'] = df['path'].apply(prediction)
df.to_csv('encoded.csv')