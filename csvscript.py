from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import librosa
import librosa.display
import os
import soundfile 
from os import listdir
from pydub import AudioSegment
import codecs
import shutil


met=0
folder="wavfiles"
for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                True
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


with codecs.open('My_voices.csv','w',encoding='utf-8') as f:
    for row in range(0,1):
        f.write('fname,label')
    path = 'TRAIN\\'
    audio_files = os.listdir(path)
    for file in audio_files:
        song = AudioSegment.from_file(f'{path}{file}')
        
        
        song.export(f"{file}.wav", format="wav", bitrate="128k")
          
        shutil.move(f"{file}.wav", f"wavfiles\\{file}{met}.wav")
        l= file.split("_")
        x = f'\n{file}{met}.wav,{l[0]}'
        f.write(x)  
        met=met+1

   
    path = 'RETRAIN\\'
    
    audio_files = os.listdir(path)
    for file in audio_files:
        song = AudioSegment.from_file(f'{path}{file}')
        
        
        song.export(f"{file}.wav", format="wav", bitrate="128k")
       
        shutil.move(f"{file}.wav", f"wavfiles\\{file}{met}.wav")
        l= file.split("_")

        x = f'\n{file}{met}.wav,{l[0]}'
        f.write(x)  
        met=met+1
