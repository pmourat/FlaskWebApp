import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
from soundfile import SoundFile
import shutil
import sys
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=1, ncols=32, sharex=False,
                             sharey=True, figsize=(20,3))
    fig.suptitle('Time Series', size=16)
    i = 0

    #x=1
  #  for x in range(2):
    for x in range(0,32):
        axes[x].set_title(list(signals.keys())[i])
        axes[x].plot(list(signals.values())[i])
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=1, ncols=32, sharex=False,
                             sharey=True, figsize=(20,3))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
   # for x in range(1):
    for x in range(0,32):
        data = list(fft.values())[i]
        Y, freq = data[0], data[1]
        axes[x].set_title(list(fft.keys())[i])
        axes[x].plot(freq, Y)
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=1, ncols=32, sharex=False,
                             sharey=True, figsize=(20,3))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    #for x in range(1):
    for x in range(0,32):
        axes[x].set_title(list(fbank.keys())[i])
        axes[x].imshow(list(fbank.values())[i],
                cmap='hot', interpolation='nearest')
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=1, ncols=32, sharex=False,
                             sharey=True, figsize=(20,3))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    #for x in range(1):
    for x in range(0,32):
        axes[x].set_title(list(mfccs.keys())[i])
        axes[x].imshow(list(mfccs.values())[i],
                cmap='hot', interpolation='nearest')
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/40), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask



df=pd.read_csv('My_voices.csv')
df.set_index('fname',inplace=True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles\\'+f)
    df.at[f, 'length'] = signal.shape[0]/rate  #length of signal divided by s rate mas dinei to length toy signal se seconds
      

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Dist', y=1.08)
ax.pie(class_dist, labels=class_dist.index,autopct="%1.1f%%",shadow=False, startangle=90)
ax.axis('equal')
#plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}
for c in classes:
    wav_file= df[df.label == c].iloc[0,0]
    signal, rate = librosa.load('wavfiles\\'+wav_file, sr=44100) #To preserve the native sampling rate of the file, use sr=None
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c]=signal
    fft[c] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=2048).T     #s1/ window(0.025)=40   nfft=sr/40 = 1102 στρογγυλοποιηση σε επομενη δυναμ του 2
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate,numcep=13, nfilt=26, nfft=2048).T
    mfccs[c] = mel
    
    
#plot_signals(signals)
#plt.show()

#plot_fft(fft)
#plt.show()

#plot_fbank(fbank)
#plt.show()

#plot_mfccs(mfccs)
#plt.show()
folder = 'clean'

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

if len(os.listdir('clean\\')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wavfiles\\'+f, sr=16000)
        mask = envelope(signal, rate, 0.005)
        wavfile.write(filename='clean\\'+f, rate=rate, data=signal[mask])

