import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from scipy.io import wavfile
import scipy
from scipy.signal import butter, lfilter, firwin


# CREA EL FILTRO PASO BAJO
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# USAR EL FILTRO PASO BAJO SOBRE LA SEÑAL
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #Aplicar filtro a la señal
    y = lfilter(b, a, data)
    return y

def getSignalTime(fs_rate, signal):
    signal_len = float(len(signal))
    tAux = float(signal_len) / float(fs_rate)
    t = np.linspace(0, tAux, signal_len)
    return t

#GRÁFICA DE LA TRANSFORMADA DE FOURIER
def plotTransform(xft, ft, title):
    #Titulo gráfico
    plt.title("Transformada " + title)
    #Ejes x e y
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.plot(xft, abs(ft))

#CALCULO DE LA TRANSFORMADA DE FOURIER
def calcFFT(fs_rate, signal):
    #Se calcula la transformada
    fft = scipy.fft(signal)
    #Normalizar transformada dividiendola por el largo de la señal
    fftNormalized = fft / len(signal)
    #Genera las frecuencias de muestreo de acuerdo al largo de fftNormalize y el inverso de la tasa de muetreo
    xfft = np.fft.fftfreq(len(fftNormalized), 1 / fs_rate)
    return xfft, fftNormalized

#GRAFICA DE SEÑAL VS TIEMPO
def plotSignalTime(signal, t, title):
    plt.plot(t, signal)
    #Titulo del gráfico
    plt.title("Amplitud [dB] vs Tiempo (s) " + title)
    #Ejes x e y
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.subplots_adjust(hspace=1)
#########################################################################################

def graph_portadora(portadora,rate):
    plt.xlabel('Tiempo (s)')
    plt.ylabel('f(t)')
    Tiempo = np.linspace(0, len(portadora) / rate, num=len(portadora))
    plt.plot(Tiempo, portadora)
    plt.xlim(0,0.015)
    plt.show()

def read_wav_file(filename):
    rate,data = wavfile.read(filename)
    return rate,data

######################################################################################################
#########################################                  ###########################################
#########################################   MODULACION AM  ###########################################
#########################################                  ###########################################
######################################################################################################

def interpolacionAM(signal,frecuencia):
    Tiempo=np.linspace(0, len(signal)/frecuencia, num=len(signal))
    funcionInterpolada = interpolate.interp1d(Tiempo, signal)
    Tiempo2 = np.linspace(0, len(signal)/frecuencia, len(signal)*4)
    y2 = funcionInterpolada(Tiempo2)
    return Tiempo2,y2

def funcionPortadoraAM(signalInterpol,rateOriginal,porcentModulacion):
    largoAM = len(signalInterpol)
    tiempoAM = np.linspace(0, largoAM / rateOriginal, largoAM)
    portadora = np.cos(2 * np.pi * rateOriginal*3 * tiempoAM)*porcentModulacion
    return portadora

def modulation_am_time(porcentModulacion):
    rate,data = read_wav_file("handel.wav")
    Tiempo,signal_interp_AM = interpolacionAM(data,rate)
    portadora = funcionPortadoraAM(signal_interp_AM,rate,porcentModulacion)
    y = signal_interp_AM * portadora
    plt.figure(1)
    plotSignalTime(y,Tiempo,"Señal AM Modulada al  %"+str(porcentModulacion*100))
    xCarryFFt, yCarryFFt = calcFFT(rate, y)
    plt.figure(2)
    plotTransform(xCarryFFt, yCarryFFt, "Señal Modulada con indice de modulación al %"+str(porcentModulacion*100))

    #graph_AM(y,rate,signal_interp_AM,portadora,"handel.wav")

def GraficoAM(signal,frecuencia,audio, filename):
    plt.xlabel('Tiempo (s)')
    plt.ylabel('f(t)')
    Tiempo = np.linspace(0, len(audio) / frecuencia, num=len(audio))
    plt.plot(Tiempo, audio, 'g', label = "Senal audio "+filename)
    plt.plot(Tiempo, signal, label = "Senal modulada")
    plt.legend()
    plt.xlim(0, 0.1)
    plt.show()

def graph_AM(signal,rate,data,portadora,filename):
    plt.ion()
    plt.figure("Portadora")
    graph_portadora(portadora,rate)
    plt.pause(0.0001)
    plt.figure("Modulacion AM")
    GraficoAM(signal, rate, data, filename)
    plt.pause(0.0001)
    plt.show(block=True)

######################################################################################################
#########################################                  ###########################################
#########################################   MODULACION FM  ###########################################
#########################################                  ###########################################
######################################################################################################

def funcionPortadoraFM(signalInterpol,rate,porcentModulacion):
    largo_FM = len(signalInterpol)
    tiempo_FM = np.linspace(0, largo_FM / rate,largo_FM)
    m2 = (np.cumsum(signalInterpol) / rate)*porcentModulacion
    aux = 2 * np.pi * rate*2 * tiempo_FM
    z = np.cos(aux + m2)
    portadora = np.cos(aux)
    return z,portadora

def modulation_fm(porcentModulacion):
    rate,data = read_wav_file("handel.wav")
    Tiempo,signal_interp_FM = interpolacionFM(data,rate)
    z, portadora = funcionPortadoraFM(signal_interp_FM,rate,porcentModulacion)
    plt.figure(3)
    plotSignalTime(z, Tiempo, "Señal AFM Modulada al  %"+str(porcentModulacion*100))
    xCarryFFt, yCarryFFt = calcFFT(rate, z)
    plt.figure(4)
    plotTransform(xCarryFFt, yCarryFFt, "Transformada Señal FM Modulada con Índice al %"+str(porcentModulacion*100))
    plt.show()
    # graph_FM(z,rate,signal_interp_FM,portadora,"handel.wav")

def interpolacionFM(signal, frecuencia):
    Tiempo = np.linspace(0, len(signal) / frecuencia, num=len(signal))
    interpolada = interpolate.interp1d(Tiempo, signal)
    Tiempo2 = np.linspace(0, len(signal) / frecuencia, len(signal) * 4)
    y2 = interpolada(Tiempo2)
    return Tiempo2, y2

def graph_FM(signal, frecuencia, audio, portadora, filename):
    plt.ion()
    plt.figure("Portadora")
    graph_portadora(portadora, frecuencia)
    plt.pause(0.0001)
    plt.figure("Modulacion FM")
    GraficoFM(signal, frecuencia, audio, filename)
    plt.pause(0.0001)
    plt.show(block=True)

def GraficoFM(signal,frecuencia,audio,filename):
     plt.xlabel('Tiempo (s)')
     plt.ylabel('f(t)')
     Tiempo=np.linspace(0, len(signal)/frecuencia, num=len(signal))
     plt.plot(Tiempo, audio, 'g', label="Senal audio " + filename)
     plt.plot(Tiempo, signal, label="Senal modulada")
     plt.legend()
     plt.xlim(0,0.10)
     plt.ylim(-10,10)
     plt.show()



modulation_am_time(1.25)
modulation_fm(1.25)
