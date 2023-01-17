import numpy as np
import librosa
import librosa.display
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import requests
from scipy.signal import savgol_filter, find_peaks,butter,hilbert, filtfilt

app = Flask(__name__)

mfccs = 0.0


def extract_data(file_name):
    print('extract start!')
    # function to load files and extract features
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # we extract mfcc feature from data
        global mfccs
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ")
        features = np.array(mfccs).reshape([-1, 1])
    print(mfccs)
    return features

def heart_rate(filename):
    print('heartrate start!')
    y, fs = librosa.load(filename, sr=40000)
    def homomorphic_envelope(y, fs, f_LPF=8, order=3):
        b, a = butter(order, 2 * f_LPF / fs, 'low')
        he = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(y)))))
        return he
    he = homomorphic_envelope(y, fs)
    x = he - np.mean(he)
    corr = np.correlate(x, x, mode='same')
    corr = corr[int(corr.size/2):]
    min_index = int(0.5*fs)
    max_index = int(2*fs)
    index = np.argmax(corr[min_index:max_index])
    true_index = index+min_index
    heartRate = 60/(true_index/fs)
    return heartRate
af = load_model('mir.h5')
murmur = load_model('murmur.h5')

app = Flask(__name__)


@app.route("/text", methods=["POST"])
def text():
    name_file = request.values["name"]
    return('hello' + name_file)


@app.route("/heart", methods=["POST"])
def audio():
    data = []
    url = request.values["url"]
    #adjust filename
    ch = '='
    listOfWords = url.split(ch, 1)
    if len(listOfWords) > 0:
        url_name = listOfWords[1]
    listOfWords = url.split(ch, 1)
    if len(listOfWords) > 0:
        url_name = listOfWords[1]
    filename = url_name+".mp3"
    print(filename)
    #download files
    response = requests.get(url)
    open(filename, "wb").write(response.content)
    #Processing
    features = extract_data(filename)
    heartRate = heart_rate(filename)
    #AI stuff
    data.append(features)
    af_result = af.predict(np.array(data))
    murmur_result = murmur.predict(np.array(data))
    y = af_result[0]
    b = murmur_result[0]
    af_return = y[0]*100
    murmur_return = b[0]*100
    returnvalue = [af_return, murmur_return,heartRate]
    print('Retrun: ', returnvalue)
    return(str(returnvalue))

if __name__ == '__main__':
    app.run()

