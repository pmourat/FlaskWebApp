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
#import glob 
import shutil
#import ffmpeg


pathb = 'patientss\\alekos\\'
audio_files = os.listdir(pathb)
for file in audio_files:
    l= file.split("_")
    song = AudioSegment.from_file(f'{pathb}{file}')
    audio_d = base64.b64encode(song)  
    cur= mysql.connection.cursor()
    cur.execute("INSERT INTO datasetwords(dtwords_name,dtwords_recording) "\
    "VALUES(%s,%s)",(l[0],audio_d, \
    ))
    mysql.connection.commit()
    cur.close()

    
   #cur= mysql.connection.cursor()
       
    #cur.execute("SELECT * FROM recordings")
    #archivo = cur.fetchall()
    #cx=session['username']
    #for f in archivo:

     #   arch=f['recording']
      #  arw=f['word_name']
       # data = base64.b64decode(arch)
       # with open(f"{arw}", "wb") as file:
        #    file.write(data)
       # file.close()
       # song = AudioSegment.from_file(f'{arw}')        
       # song.export(f"{arw}.wav", format="wav", bitrate="128k")   
       # shutil.move(f"{arw}.wav", f"patients//{cx}//{arw}.wav")


