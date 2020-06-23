import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from scipy.io import wavfile
import scipy
from scipy.signal import butter, lfilter


#Filtro paso bajo
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

#Aplicación del filtro de paso bajo
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #Aplicar filtro a la señal
    y = lfilter(b, a, data)
    return y

#Función que permite obtener un vector de tiempo en base a una señal
def getSignalTime(fs_rate, signal):
    signal_len = float(len(signal))
    tAux = float(signal_len) / float(fs_rate)
    t = np.linspace(0, tAux, signal_len)
    return t

#Función que permite graficar la tranformada de Fourier de una señal
def plotTransform(ejeXTransformada, ejeYTransformada, Titulo):
    plt.title("Transformada " + Titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.plot(ejeXTransformada, abs(ejeYTransformada))

#Función que permite calcular la transformada de Fourier de una señal.
def calcFFT(fs_rate, signal):
    #Calculo de transformada
    fft = scipy.fft(signal)
    #Normalización
    fftNormalized = fft / len(signal)
    #Generación de frecuencias de muestreo
    xfft = np.fft.fftfreq(len(fftNormalized), 1 / fs_rate)
    return xfft, fftNormalized

#Función que permite graficar una señal en el tiempo
def plotSignalTime(data, tiempo, titulo):
    plt.plot(tiempo, data)
    plt.title("Amplitud [dB] vs Tiempo (s) " + titulo)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.subplots_adjust(hspace=1)

#Función que permite extraer tanto los datos de un audio y su frencuencia
def read_wav_file(filename):
    rate,data = wavfile.read(filename)
    return rate,data

######################################################################################################
#########################################                  ###########################################
#########################################   MODULACION AM  ###########################################
#########################################                  ###########################################
######################################################################################################

#Funcion que permite interpolar una señal AM en base a la señal y su frecuencia
def interpolacionAM(signal,frecuencia):
    #Se genera vector de tiempo
    Tiempo=getSignalTime(frecuencia,signal)
    #Uso de funcion interpolar de scipy, la cual retorna una funcion que puede ser aplicada a un vector
    funcionInterpolada = interpolate.interp1d(Tiempo, signal)
    Tiempo2 = np.linspace(0, len(signal)/frecuencia, len(signal)*4)
    y2 = funcionInterpolada(Tiempo2)
    return Tiempo2,y2

#Funcion que permite generar una funcion portadora en base a un indice de modulacion
def funcionPortadoraAM(signalInterpol,rateOriginal,porcentModulacion):
    largoAM = len(signalInterpol)
    tiempoAM = np.linspace(0, largoAM / rateOriginal, largoAM)
    #Se crea la funcion portadora para una señal AM
    portadora = np.cos(2 * np.pi * rateOriginal*3 * tiempoAM)*porcentModulacion
    return portadora

#Funcion que permite modular una señal AM en base a un indice de modulacion
def modulation_am_time(porcentModulacion,nFigura):
    rate,data = read_wav_file("handel.wav")
    Tiempo,signal_interp_AM = interpolacionAM(data,rate)
    portadora = funcionPortadoraAM(signal_interp_AM,rate,porcentModulacion)
    y = signal_interp_AM * portadora
    plt.figure(nFigura)
    plotSignalTime(y,Tiempo,"Señal AM Modulada al  %"+str(porcentModulacion*100))
    xCarryFFt, yCarryFFt = calcFFT(rate, y)
    plt.figure(nFigura+1)
    plotTransform(xCarryFFt, yCarryFFt, "Señal Modulada con indice de modulación al %"+str(porcentModulacion*100))
    return y,Tiempo,rate,data


######################################################################################################
#########################################                  ###########################################
#########################################   MODULACION FM  ###########################################
#########################################                  ###########################################
######################################################################################################

#Funcion que permite generar una funcion portadora en base a un indice de modulacion
def funcionPortadoraFM(signalInterpol,rate,porcentModulacion):
    largo_FM = len(signalInterpol)
    tiempo_FM = np.linspace(0, largo_FM / rate,largo_FM)
    m2 = (np.cumsum(signalInterpol) / rate)*porcentModulacion
    aux = 2 * np.pi * rate*2 * tiempo_FM
    z = np.cos(aux + m2)
    # Se crea la funcion portadora para una señal FM
    portadora = np.cos(aux)
    return z,portadora

#Funcion que permite modular una señal FM en base a un indice de modulacion
def modulation_fm(porcentModulacion,nFigura):
    rate,data = read_wav_file("handel.wav")
    Tiempo,signal_interp_FM = interpolacionFM(data,rate)
    z, portadora = funcionPortadoraFM(signal_interp_FM,rate,porcentModulacion)
    plt.figure(nFigura)
    plotSignalTime(z, Tiempo, "Señal AFM Modulada al  %"+str(porcentModulacion*100))
    xCarryFFt, yCarryFFt = calcFFT(rate, z)
    plt.figure(nFigura+1)
    plotTransform(xCarryFFt, yCarryFFt, "Transformada Señal FM Modulada con Índice al %"+str(porcentModulacion*100))

#Funcion que permite interpolar una señal FM en base a la señal y su frecuencia
def interpolacionFM(signal, frecuencia):
    # Se genera vector de tiempo
    Tiempo = getSignalTime(frecuencia,signal)
    # Uso de funcion interpolar de scipy, la cual retorna una funcion que puede ser aplicada a un vector
    interpolada = interpolate.interp1d(Tiempo, signal)
    Tiempo2 = np.linspace(0, len(signal) / frecuencia, len(signal) * 4)
    y2 = interpolada(Tiempo2)
    return Tiempo2, y2


####################################         MAIN        #########################################
#Se aplica modularizacion AM
y,Tiempo,rate,data= modulation_am_time(0.15,1)
y,Tiempo,rate,data= modulation_am_time(1,5)
y,Tiempo,rate,data= modulation_am_time(1.25,9)

#Se aplica modularizacion FM
modulation_fm(0.15,3)
modulation_fm(1,7)
modulation_fm(1.25,11)

#Se aplica demodularizacion
plt.figure(13)
TiempoOriginal = getSignalTime(rate,data)
plotSignalTime(data, TiempoOriginal, "Señal Original")

plt.figure(14)
#Aplicacion de demodularizacion de la señal modularizada por Portadora AM usando un filtro paso bajo
demodulated = butter_lowpass_filter(y, 2000, rate)
plotSignalTime(demodulated, Tiempo, "Señal Demodulada")

plt.show()