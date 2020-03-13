import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import os
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config
from keras.optimizers import SGD

import ctypes

#hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudart64_100.dll")


def check_data():
    if os.path.isfile(config.p_path):
        print("LOADING EXISTING DATA FOR {} MODEL".format(config.mode))
        with open(config.p_path, "rb") as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None



def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(int(n_samples))):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean\\'+file)
        label = df.at[file, 'label']
        if((wav.shape[0]-config.step)<0):
            print(wav.shape[0])
            print(config.step) 
            continue
        
        rand_index = np.random.randint(0, abs(wav.shape[0]-config.step))
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        _max = max(np.amax(X_sample), _max)
        _min = min(np.amin(X_sample), _min)                
        X.append(X_sample)
        y.append(classes.index(label))

    config.min = _min
    config.max = _max    
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=mikos) #allazei analoga me to posses klaseis exoyme
    config.data = (X,y)

    with open(config.p_path, "wb") as handle:
        pickle.dump(config, handle, protocol=2)
    return X, y  

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation="relu", strides=(1,1),padding='same'))
    model.add(Conv2D(64, (3,3), activation="relu", strides=(1,1),padding='same'))
    model.add(Conv2D(128, (3,3), activation="relu", strides=(1,1),padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(mikos, activation='softmax')) # gia tis klasseis mas
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


folder="models"
for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                True
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


folder="pickles"
for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                True
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))            

df = pd.read_csv('My_voices.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean\\'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
mikos=len(classes)
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2*int(df['length'].sum())/0.1   
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)





config = Config(mode='conv')

X, y = build_rand_feat()
y_flat = np.argmax(y, axis=1) #gia na boroyme na epanerthoyme sta kanonika classes names.epd tr ta antistoixisame se 0 kai 1
input_shape = (X.shape[1], X.shape[2], 1)
model = get_conv_model()

class_weight = compute_class_weight("balanced",
                                    np.unique(y_flat),y_flat)


checkpoint = ModelCheckpoint(config.model_path, monitor=("val_loss" + "val_acc"), verbose=1, mode="max",save_best_only=True, save_weights_only=False, period=1)

model.fit(X, y, epochs=7, batch_size=32, shuffle=True, validation_split=0.1, callbacks=[checkpoint])

model.save(config.model_path)
#model.save('Audio_classification.h5')