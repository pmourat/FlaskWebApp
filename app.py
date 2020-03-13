from __future__ import print_function

from flask import Flask, render_template, request, redirect, session, flash , url_for
from flask_bootstrap import Bootstrap
from flask_mysqldb import MySQL
import yaml
import zipfile
import base64
import io
import pathlib
from flask import abort
from flask import make_response
from flask import redirect
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CsrfProtect
import os
import os.path
import uuid
import matplotlib.pyplot as plt
import csv
import librosa
import librosa.display
import os
import soundfile 
from os import listdir
from distutils.dir_util import copy_tree
from os import path
from pydub import AudioSegment
import codecs
 
import shutil
import flask as fl
#import ffmpeg
import pickle
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from keras.models import load_model
import pandas as pd 
from sklearn.metrics import accuracy_score
from python_speech_features import mfcc, logfbank
from soundfile import SoundFile
import keras.backend.tensorflow_backend as tb
import time
import datetime
import re



tb._SYMBOLIC_SCOPE.value = True
app = Flask(__name__)
Bootstrap(app)

db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

app.config['SECRET_KEY'] = 'secret'

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")



@app.route('/about/')
def about():

    return render_template('about.html')

@app.route('/patients/<int:id>')
def patients(id):
    cur = mysql.connection.cursor()
    sd = cur.execute("SELECT * FROM results WHERE patient_id = {}".format(id))
    ret=cur.fetchall()
    
       

    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT * FROM patient WHERE patient_id = {}".format(id))

    if resultValue > 0:
        patient = cur.fetchone()
        return render_template('patient.html', patient=patient,ret=ret,sd=sd)
    return 'Patient not found'

@app.route('/tests/<int:id>')
def tests(id):
    cur = mysql.connection.cursor()
    result = cur.execute("SELECT * FROM words")
    if result > 0:
        word = cur.fetchall()
    resultValue = cur.execute("SELECT * FROM test WHERE test_id = {}".format(id))
    
    if resultValue > 0:
        test = cur.fetchone()
        return render_template('test.html', test=test,word=word)
    return 'Test not found'

@app.route('/create_test/', methods=['GET', 'POST'])
def create_test():

        
    if request.method == 'POST':

        testDetails=request.form
        cur= mysql.connection.cursor()
        cur.execute("INSERT INTO test(test_name) "\
        "VALUES(%s)",(testDetails['test_name'], \
        ))
        mysql.connection.commit()
        cur.close()
        cur= mysql.connection.cursor()
        x= testDetails['test_name']
        
        result_value = cur.execute("SELECT * FROM test WHERE test_name = %s",[x])

        result=cur.fetchone()
        cur.close()
        cur= mysql.connection.cursor()

        for i in range(32):
            f=str (i)
            f="word"+f
            if (testDetails[f] != ""):
                cur.execute("INSERT INTO words(test_name,word_name,test_id) "\
                "VALUES(%s,%s,%s)",(testDetails['test_name'],testDetails[f],result['test_id'] \
                ))

                

        mysql.connection.commit()
        cur.close()
        flash('Test created successfully!', 'success')
        return redirect('/create_test')

    return render_template('create_test.html')




@app.route('/download-zip/<int:id>/', methods=['GET','POST'])
def request_zip(id):
    h='patient//'
    cur = mysql.connection.cursor()
    result_value = cur.execute("SELECT * FROM patient WHERE patient_id = {}".format(id))
    if result_value > 0:
        patient = cur.fetchone()
        patient_form = {}
        patient_form['username'] = patient['username']
        s=patient_form['username']
    
    if os.path.isdir(f"patients//{s}"):
        filee = f"patients//{s}Predictions.csv"   
        
        if os.path.exists(filee):
            ses='patient//' + s +'//'

            if not os.path.exists(ses):
                os.makedirs(ses)
            else:    
                for filename in os.listdir(ses):
                    file_path = os.path.join(ses, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            True
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e)) 

            shutil.move(f'patients//{s}Predictions.csv', f'{h}{s}')
            fromDirectory = f"patientss//{s}"
            toDirectory = f"patient//{s}"

            copy_tree(fromDirectory, toDirectory)



           


        base_path = pathlib.Path(f"./patient\\{s}")
        data = io.BytesIO()
    
        with zipfile.ZipFile(data, mode='w') as z:
            for f_name in base_path.iterdir():
                z.write(f_name)
        data.seek(0)
        

        return fl.send_file(
            data,
            mimetype='application/zip',
            as_attachment=True,
            attachment_filename='data.zip',
            cache_timeout=0
        )
    else:
        cur = mysql.connection.cursor()
        result_value = cur.execute("SELECT * FROM patient WHERE name != 'therapist'")
        if result_value > 0:
            my_patients = cur.fetchall()
            flash('Test not done yet', 'danger')
            return render_template('my_patients.html',my_patients=my_patients)
        else:
            flash('Test not done yet', 'danger')
            return render_template('my_patients.html',my_patients=None)

        
   




@app.route("/create/", methods=['GET','POST'])
def create():
    
    cur = mysql.connection.cursor()
    result = cur.execute("SELECT * FROM test")    

  
    my_test=cur.fetchall() 
    if request.method == 'POST':
        userDetails = request.form
        if userDetails['password'] != userDetails['confirm_password']:
            flash('Passwords do not match! Try again.', 'danger')
            return render_template('create_patient.html',my_test=my_test)

        cur = mysql.connection.cursor()
        
        cur.execute("INSERT INTO patient(name, surname, username, password, email,description,problem,age) "\
        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",(userDetails['first_name'], userDetails['last_name'], \
        userDetails['username'], userDetails['password'], userDetails['email'], userDetails["description"], userDetails["optradio"],userDetails['age']))
        mysql.connection.commit()
        cur.close()
        flash('Registration successful!', 'success')
        return redirect('/')
    return render_template('create_patient.html',my_test=my_test)

@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userDetails = request.form
        username = userDetails['username']
        cur = mysql.connection.cursor()
        resultValue = cur.execute("SELECT * FROM patient WHERE username = %s", ([username]))
        if resultValue > 0:
            user = cur.fetchone()
            if ((userDetails['password'] == "therapist") and (userDetails['password'] == user['password'])):
                session['login'] = True
                session['firstName'] = user['name']
                session['lastName'] = user['surname']
                flash('Welcome ' + session['firstName'] +'! You have been successfully logged in', 'success')
            elif userDetails['password'] == user['password']:
                #REDIRECT ΣΤΟΝ ΗΧΟΓΡΑΦΗΤΗ
                session['login'] = True
                session['firstName'] = user['name']
                session['lastName'] = user['surname']
                session['username'] = user['username']
                session['age'] = user['age']
                onoma=user['problem']
                p =cur.execute("SELECT word_name FROM words WHERE test_name = %s", ([onoma])) #plithos
                session['p']=p
                w=cur.fetchall()
                session['words']=w
                l=[]
                for f in w:
                    l.append((f['word_name']))
                    
                session['words']=l 

                return render_template("welcome.html",onoma = onoma,w=w)
                flash('Welcome ' + session['firstName'] +'! You have been successfully logged in', 'success')
            else:
                cur.close()
                flash('Password does not match', 'danger')
                return render_template('login.html')
        else:
            cur.close()
            flash('User not found', 'danger')
            return render_template('login.html')
        cur.close()
        return redirect('/')
    return render_template('login.html')



@app.route('/my-patients/')
def view_patients():
    name = session['firstName']
    cur = mysql.connection.cursor()
    result_value = cur.execute("SELECT * FROM patient WHERE name != 'therapist'")
    if result_value > 0:
        my_patients = cur.fetchall()
        return render_template('my_patients.html',my_patients=my_patients)
    else:
        return render_template('my_patients.html',my_patients=None)


@app.route('/my-tests/')
def view_tests():
    cur = mysql.connection.cursor()
    result_value = cur.execute("SELECT * FROM test ")
    if result_value > 0:
        my_tests = cur.fetchall()
        return render_template('my_tests.html',my_tests=my_tests)
    else:
        return render_template('my_tests.html',my_tests=None)        


@app.route('/edit-patient/<int:id>/', methods=['GET', 'POST'])
def edit_patient(id):

    
    cur = mysql.connection.cursor()
    result = cur.execute("SELECT * FROM test")    
    my_test=cur.fetchall() 

    if request.method == 'POST':
        cur = mysql.connection.cursor()
        name = request.form['first_name']
        surname = request.form['last_name']
        password = request.form['password']
        username = request.form['username']
        age = request.form['age']

        email = request.form['email']
        description = request.form['description']
        problem = request.form['optradio']
        

        cur.execute("UPDATE patient SET name = %s, surname = %s, password = %s, username = %s,email = %s, description = %s, problem = %s, age =%s where patient_id = %s",(name,surname,password,username,email,description,problem,age, id))
        mysql.connection.commit()
        cur.close()
        flash('Patient updated successfully', 'success')
        return redirect('/patients/{}'.format(id))
    cur = mysql.connection.cursor()
    result_value = cur.execute("SELECT * FROM patient WHERE patient_id = {}".format(id))
    if result_value > 0:
        patient = cur.fetchone()
        patient_form = {}
        patient_form['name'] = patient['name']
        patient_form['surname'] = patient['surname']
        patient_form['password'] = patient['password']
        patient_form['username'] = patient['username']
        patient_form['email'] = patient['email']
        patient_form['description'] = patient['description']
        patient_form['problem'] = patient['problem']
        patient_form['age'] = patient['age']
        

        return render_template('edit_patient.html', patient_form=patient_form, my_test=my_test)


@app.route('/edit-test/<int:id>/', methods=['GET', 'POST'])
def edit_test(id):

    testDetails = request.form
    cur = mysql.connection.cursor()
    res = cur.execute("SELECT * FROM words WHERE test_id = {}".format(id))    

    my_words=cur.fetchall()   
    b = cur.execute("SELECT COUNT(word_name) FROM words WHERE test_id = {}".format(id)) 
    c= cur.fetchone()
    i=c['COUNT(word_name)']

    if request.method == 'POST':
        cur = mysql.connection.cursor()
        test_name = request.form['test_name']
              

        cur.execute("UPDATE test SET test_name = %s WHERE test_id = %s",(test_name, id))
        mysql.connection.commit()
        cur.close()

        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM words WHERE test_id = {}".format(id))
        



        mysql.connection.commit()
        cur.close()
        cur = mysql.connection.cursor()


        for i in range(32):
            f=str (i)
            f="word"+f
            if (testDetails[f] != ""):
                cur.execute("INSERT INTO words(test_name,word_name,test_id) "\
                "VALUES(%s,%s,%s)",(testDetails['test_name'],testDetails[f],id \
                ))



        mysql.connection.commit()
        cur.close()        


     
        flash('Test updated successfully', 'success')
        return redirect('/tests/{}'.format(id))


    cur = mysql.connection.cursor()
    result_value = cur.execute("SELECT * FROM test WHERE test_id = {}".format(id))
    if result_value > 0:
        test = cur.fetchone()
        test_form = {}
        test_form['test_name'] = test['test_name']
        test_form['test_id'] = test['test_id']


        return render_template('edit_test.html', test_form=test_form,my_words=my_words, i=i)




@app.route('/delete-patient/<int:id>/')
def delete_patient(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM patient WHERE patient_id = {}".format(id))
    mysql.connection.commit()
    flash("Patient has been deleted", 'success')
    return redirect('/my-patients')

@app.route('/delete-test/<int:id>/')
def delete_test(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM test WHERE test_id = {}".format(id))
    mysql.connection.commit()
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM words WHERE test_id = {}".format(id))
    mysql.connection.commit()
    
    flash("Test has been deleted", 'success')
    return redirect('/my-tests')    


@app.route('/logout/')
def logout():
    session.clear()
    flash("You have been logged out", 'info')
    return redirect('/')

#ΗΧΟΓΡΑΦΗΤΗΣ

@app.route("/speech",methods=['GET','POST'])
def welcome():
   
    session_id = request.cookies.get('session_id')
    if session_id:
        all_done = request.cookies.get('all_done')
        if all_done:

            return redirect(url_for('runn'))

            return render_template("thanks.html")

        else:
            return render_template("record.html")
    else:
        return render_template("welcome.html")


@app.route("/start",methods=['GET','POST'])
def start():
    response = make_response(redirect('/speech'))
    session_id = uuid.uuid4().hex
    response.set_cookie('session_id', session_id)
    return response


@app.route('/upload', methods=['GET','POST'])
def upload():
    session_id = request.cookies.get('session_id')
    if not session_id:
        make_response('No session', 400)

    word = request.args.get('word')
    audio_data = request.data
    usr=session['username']
    filename = word + '_' + session['username'] + '_' + session_id + '_' + uuid.uuid4().hex + '.ogg'
    cur= mysql.connection.cursor()
    cur.execute("SELECT * FROM patient WHERE username =  %s",[usr])
    pat = cur.fetchone()
    idpat=pat['patient_id']
    idsr=pat['problem']
    cur= mysql.connection.cursor()
    cur.execute("SELECT * FROM words WHERE test_name =  %s",[idsr])
    sv= cur.fetchone()
    ssf=sv['test_name']
    s='patients//' + session['username'] +'//'

    if not os.path.exists(s):
      os.makedirs(s)

    with open(os.path.join(s,filename), "wb") as file1:

        file1.write(audio_data)
        audio_d = base64.b64encode(audio_data)

    
    cur= mysql.connection.cursor()
    cur.execute("INSERT INTO recordings(recording,word_name,patient_id,test_name,patient_name) "\
    "VALUES(%s,%s,%s,%s,%s)",(audio_d,filename,idpat,ssf,usr, \
    ))
    mysql.connection.commit()
    cur.close()


    
    return make_response('All good')




@app.route('/retrainm',methods=['GET','POST'])
def retrann():
        

        os.system("csvscript.py")
        os.system("dataprep.py")
        os.system("model.py")
        flash('Model retrained successfully!', 'success')

        return redirect('/')



        
        



@app.route('/retr',methods=['GET','POST'])
def retra():
        
        return render_template('ret.html')
    

















@app.route('/run',methods=['GET','POST'])
def runn():

    



    # Για εξαγωγή ηχογραφήσεων από τον πίνακα recordings
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



    session_id = request.cookies.get('session_id')

    
    s='patients//'

    b=session['username']
  
    with codecs.open('My_voices.csv','a',encoding='utf-8') as f:
    
    
        path ='patients//' + b + "//"
        if not os.path.exists(path):

             os.makedirs(path)

        folder = 'patients//' + b + f"//{b}Predictions.csv"   
        
        if os.path.exists(folder):
            os.remove(folder)

        audio_files = os.listdir(path)
        
        n=session['words']
        for file in audio_files:
            song = AudioSegment.from_file(f'{path}{file}')
        
        
            song.export(f"{file}.wav", format="wav", bitrate="128k")
          
            shutil.move(f"{file}.wav", f"test\\{file}.wav")
            
            l= file.split("_")

            x = f'\n{file}.wav,{l[0]}'
            f.write(x)
            
    with codecs.open(f'{b}.csv','w',encoding='utf-8') as f:
        for row in range(0,1):
            f.write('fname,label')
        path ='patients//' + b + "//"
        if not os.path.exists(path):

             os.makedirs(path)

        audio_files = os.listdir(path)
        
        n=session['words']
        for file in audio_files:
            song = AudioSegment.from_file(f'{path}{file}')
        
        
            song.export(f"{file}.wav", format="wav", bitrate="128k")
          
            shutil.move(f"{file}.wav", f"test\\{file}.wav")
            
            l= file.split("_")

            x = f'\n{file}.wav,{l[0]}'
            f.write(x)









    def calc_fft(y, rate):
        n = len(y)
        freq = np.fft.rfftfreq(n, d=1/rate)
        Y = abs(np.fft.rfft(y)/n)
        return (Y, freq)
    

    def envelope(y, rate, threshold):
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    df=pd.read_csv(f'{b}.csv')
    
    df.set_index('fname',inplace=True)

    for f in df.index:
        rate, signal = wavfile.read('test\\'+f)
        df.at[f, 'length'] = signal.shape[0]/rate 
    if len(os.listdir('test') ) == 0:
        return render_template('thanks.html')
    else:
        r=session['username']

        folder=f"patientss//{r}"
        if os.path.isdir(folder):
            

            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        True
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    
 
    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()
    
 
    df.reset_index(inplace=True)



    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    for c in classes:
        wav_file= df[df.label == c].iloc[0,0]
        signal, rate = librosa.load('test\\'+wav_file, sr=44100) #To preserve the native sampling rate of the file, use sr=None
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals[c]=signal
        fft[c] = calc_fft(signal, rate)

        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=2048).T     #s1/ window(0.025)=40   nfft=sr/40 = 1102 στρογγυλοποιηση σε επομενη δυναμη του 2
        fbank[c] = bank
        mel = mfcc(signal[:rate], rate,numcep=13, nfilt=26, nfft=2048).T
        mfccs[c] = mel
    
    if len(os.listdir('cleantest\\')) == 0:
        for f in tqdm(df.fname):
            signal, rate = librosa.load('test\\'+f, sr=16000)
            mask = envelope(signal, rate, 0.005)
            wavfile.write(filename='cleantest\\'+f, rate=rate, data=signal[mask])


    def build_predictions(audio_dir):
        y_true = []
        y_pred = []
        fn_prob = {}

        print("Extracting features from audio")
        for fn in tqdm(os.listdir(audio_dir)):
            rate,wav = wavfile.read(os.path.join(audio_dir, fn))
            label = fn2class[fn]
            c = classes.index(label)
            y_prob = []

            for i in range(0, wav.shape[0]-config.step, config.step):
                sample = wav[i:i+config.step]
                x = mfcc(sample, rate, numcep=config.nfeat,nfilt=config.nfilt, nfft=config.nfft)
                x = (x - config.min) / (config.max - config.min)

                x = x.reshape(1, x.shape[0], x.shape[1], 1)
                y_hat = model.predict(x)
                y_prob.append(y_hat)
                y_pred.append(np.argmax(y_hat))
                y_true.append(c)

            fn_prob[fn] = np.mean(y_prob, axis=0).flatten()

        return y_true, y_pred, fn_prob        


    df = pd.read_csv('My_voices.csv')
    
    classes = list(np.unique(df.label))
    df = pd.read_csv(f'{b}.csv')
    fn2class = dict(zip(df.fname, df.label))
    p_path = os.path.join("pickles\\", "conv.p")
    tb._SYMBOLIC_SCOPE.value = True

    with open(p_path, "rb") as handle:
        config = pickle.load(handle)

    model = load_model(config.model_path)

    y_true, y_pred, fn_prob = build_predictions("cleantest\\")
    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

    y_probs = []
    for i, row in df.iterrows():
        y_prob = fn_prob[row.fname]
        y_probs.append(y_prob)
        for c, p in zip(classes, y_prob):
            df.at[i, c] = p
    
    
    y_pred = [classes[np.argmax(y)] for y in y_probs]    
    df["y_pred"] = y_pred
    r=session['username']

    df.to_csv(f'{s}//{r}Predictions.csv', index=False)
    cur = mysql.connection.cursor()
    rest = cur.execute("SELECT patient_id FROM patient WHERE username = '{}'".format(r))
    
    idd = cur.fetchone()
    myDF=pd.read_csv(f'{s}//{r}Predictions.csv')

    newDF=myDF[['label','y_pred']]
    zewDF=myDF[['label','y_pred']]

    newDF=str(newDF)
    newDF = newDF.replace("label", "Λέξη,")
    newDF = newDF.replace("y_pred", "Πρόβλεψη")
    

      
    source = f'{s}//{r}Predictions.csv'
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    cur= mysql.connection.cursor()
    rest = cur.execute("SELECT patient_id FROM results WHERE patient_name = '{}'".format(r))
    if rest>0:
        mDF={}
        cur.execute("DELETE FROM results WHERE patient_name = '{}'".format(r))
        
        for row in zewDF.itertuples():
            mDF=(row.label,row.y_pred)
            print(row.label)

            print(row.y_pred)
                
            cur.execute("INSERT INTO results(patient_name,patient_id,timestamp,p1,p2) "\
            "VALUES(%s,%s,%s,%s,%s)",(r,idd['patient_id'],timestamp, mDF[0], mDF[1], \
            ))
       
    else:
        mDF={}
        for row in zewDF.itertuples():
            mDF=(row.label,row.y_pred)
            cur.execute("INSERT INTO results(patient_name,patient_id,timestamp,p1,p2) "\
            "VALUES(%s,%s,%s,%s,%s)",(r,idd['patient_id'],timestamp, mDF[0], mDF[1], \
            ))
    mysql.connection.commit()
    cur.close()


    



    folder = 'cleantest'

    source = 'cleantest//'
    dest1 = 'RETRAIN'

    files = os.listdir(source)

  

    folder = 'cleantest'                        
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    

    folder = 'test'

    source = 'test//'
    dest1 = 'RETRAIN'

    files = os.listdir(source)

    for f in files:
        filee = f"RETRAIN//{f}"   
        if os.path.exists(filee):
            True
        else:    
            shutil.move(f"{source}{f}", dest1)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
            
                True
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    filePathd = f'{r}.csv'
 
    if os.path.exists(filePathd):
        os.remove(filePathd)


    fromDirectory = f"patients//{r}"
    toDirectory = f"patientss//{r}"

    copy_tree(fromDirectory, toDirectory)

    folder=f"patients//{r}"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                True
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))




  
    return render_template('thanks.html')






if __name__ == '__main__':
    app.run(debug=True, port=5001,threaded=False)
