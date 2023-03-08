import soundfile as sf
import matplotlib.pyplot as plt
from ss_utils.filter_and_sampling import downsampling_signal
from heart_lung_separation import nmf_lung_heart_separation
from PyEMD import EEMD
import librosa
import numpy as np
import librosa.display
from scipy.signal import savgol_filter, hilbert, filtfilt, butter, find_peaks

def heart_rate(y, fs):
    """ def homomorphic_envelope(y, fs, f_LPF=8, order=3):
        b, a = butter(order, 2 * f_LPF / fs, 'low')
        he = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(y)))))
        return he
    he = homomorphic_envelope(y, fs)
    signal_filtered = (np.abs(he - np.mean(he))*10)**2 """
    ab_signal = np.abs(y)*5
    signal_filtered = np.abs(savgol_filter(ab_signal, 4999, 4))
    x = (sum(signal_filtered)/len(signal_filtered))*1
    peak, _ = find_peaks(signal_filtered, prominence=x)
    s1 = peak[::2]
    s2 = peak[1::2]
    """ he = homomorphic_envelope(y, fs)
    x = he - np.mean(he)
    corr = np.correlate(x, x, mode='full')
    corr = corr[int(corr.size/2):]
    min_index = int(0.5*fs)
    max_index = int(2*fs)
    index = np.argmax(corr[min_index:max_index])
    true_index = index+min_index
    heartRate = 60/(true_index/fs)
    plt.plot(y)
    plt.plot(he, linewidth=2)
    #plt.show()
    print(heartRate) """
    print(s1)
    plt.vlines(x=s1,ymin = 0, ymax = 100000, color="g")
    plt.vlines(x=s2,ymin = 0, ymax = 100000, color="r")
    plt.legend(['s1','s2'])
    plt.plot(y)
    plt.plot(signal_filtered)
    plt.axhline(y = x)
    return peak


## Módulo de testeo
if __name__ == '__main__':
    # Abriendo audio de ejemplo
    filename = 'a0007.wav'
    audio, samplerate = sf.read(filename)
    
    # Obteniendo las señales
    lung_signal, heart_signal = \
        nmf_lung_heart_separation(audio, samplerate, 
                                  model_name='definitive_segnet_based')
    
    # Aplicando downsampling
    new_rate, audio_dwns = \
                downsampling_signal(audio, samplerate, 
                                    freq_pass=11025//2-100, 
                                    freq_stop=11025//2)
    print('Nueva tasa de muestreo para plot:', new_rate)
    print(heart_rate(heart_signal, new_rate))
    plt.show()
