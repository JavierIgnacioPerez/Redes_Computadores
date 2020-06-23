import numpy as np
import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt

##################################IMPORTAR SEÑAL DE AUDIO#################################

rate, sound = scipy.io.wavfile.read('handel.wav')

#####################CREAR VECTOR TIEMPO PARA GRAFICAR PUNTOS EN GRAFICA##################

time = np.linspace(0,len(sound)/rate,len(sound))

###############################APLICACION TRANSFORMADA DE FOURIER#########################

fourier = np.fft.fft(sound)
deltaTime = 1/rate
frequencies = np.fft.fftfreq(len(sound),deltaTime)

###################################FILTROS################################################

##########################################################################################
########################## Calculo Paso Bajo  ############################################
##########################################################################################

def lowFilter(fs,corte,signalA):
    mitad = fs/2
    lowcut = corte/mitad
    b, a = signal.butter(N=3,Wn=lowcut, btype='low')
    filteredSignal=signal.filtfilt(b, a, signalA)
    len_signal = len(signalA)
    len_signal = float(len_signal)
    time = float(len_signal)/float(fs)
    rate = np.arange(0,time,1.0/float(fs))
    return rate,filteredSignal

#########################################################################################
########################## Calculo Paso Alto  ###########################################
#########################################################################################
def highFilter(fs,corte,signalA):
    mitad = fs/2
    highcut = corte/mitad
    numtaps = corte + 1
    b, a = signal.butter(N=3, Wn=highcut, btype='high')
    filteredSignal = signal.filtfilt(b, a, signalA)
    len_signal = len(signalA)
    len_signal = float(len_signal)
    time = float(len_signal)/float(fs)
    rate = np.arange(0,time,1.0/float(fs))
    return rate,filteredSignal

########################################################################################
########################## Calculo Paso Banda  #########################################
########################################################################################
def passBandFilter(fs,low,high,signalA):
    mitad = fs/2
    lowcut = low/mitad
    highcut = high/mitad
    b, a = signal.butter(N=5, Wn=[lowcut,highcut], btype='band')
    filteredSignal = signal.filtfilt(b, a, signalA)
    len_signal = len(signalA)
    len_signal = float(len_signal)
    time = float(len_signal)/float(fs)
    rate = np.arange(0,time,1.0/float(fs))
    return rate,filteredSignal


###################### Graficar Señal En El Tiempo ######################################

def plotSignalTime(sound,tiempo,title):
    plt.plot(tiempo,sound)
    plt.title("Amplitud vs tiempo "+title)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")

###################### Calcular Transformada de Fourier a una señal #####################

def calcFFT(fs,signal):
    fft = scipy.fft(signal)
    fftNormalizado = fft/len(signal)
    frecFft = np.fft.fftfreq(len(fftNormalizado),1/fs)
    return frecFft, fftNormalizado

##################### Graficar Señal En Dominio De La Frecuencia #########################

def plotTransform(xft,ft,title):
    plt.title("Transformada "+title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.plot(xft,abs(ft))

################### Graficar Espectograma De Una Señal ###################################

def plotSpec(sound,fs,title):
    plt.title("Espectrograma "+title)
    plt.specgram(sound,Fs=fs)

################## Calcular Inversa De La Transforma De Fourier ##########################

def invTransform(ft,lenFreq):
    inversaFourier = scipy.ifft(ft)*lenFreq
    return inversaFourier



#######################    MAIN      #####################################################


######################### Grafico señal de audio en el tiempo ############################

plt.figure(1)
plt.subplot(311)
plotSignalTime(sound,time,"Señal de Audio")

############# Grafico señal de Audio en Dominio de la Frecuencia #########################

plt.subplot(312)
plotTransform(frequencies,fourier,"Señal de Audio")

################### Espectograma de la señal de Audio ####################################

plt.subplot(313)
frequencies, times, spectrogram = signal.spectrogram(sound, rate)
plt.pcolormesh(times, frequencies, np.log(spectrogram))
plt.colorbar()
plt.title("Espectograma de la Función")
plt.xlabel("Tiempo [S]")
plt.ylabel("Frecuencia [Hz]")
plt.tight_layout()


################### Grafico Audio Tras Aplicar Filtro Paso Alto ####################

plt.figure(2)
title_passHigh = "Filtro Paso Alto"
plt.subplot(311)
xFilteredHigh,filteredSignalHigh = highFilter(rate,3000,sound)
plotSignalTime(filteredSignalHigh, xFilteredHigh, title_passHigh)


########## Grafico De La Transformada De Audio Tras Aplicar Filtro Paso Alto ########

xPassHigh, fftPassHigh = calcFFT(rate, filteredSignalHigh)
plt.subplot(312)
plotTransform(xPassHigh, fftPassHigh, title_passHigh)

############### Espectograma Del Audio Tras Aplicar Filtro Paso Alto ################

plt.subplot(313)
plotSpec(filteredSignalHigh, rate, title_passHigh)
plt.colorbar()
plt.tight_layout()


################### Grafico Audio Tras Aplicar Filtro Paso Bajo ####################

plt.figure(3)
title_passLow = "Filtro Paso Bajo"
plt.subplot(311)
xFilteredLow,filteredSignalLow = lowFilter(rate,1000,sound)
plotSignalTime(filteredSignalLow, xFilteredLow, title_passLow)

########## Grafico De La Transformada De Audio Tras Aplicar Filtro Paso Bajo ########

xPassLow, fftPassLow = calcFFT(rate, filteredSignalLow)
plt.subplot(312)
plotTransform(xPassLow, fftPassLow, title_passLow)


############### Espectograma Del Audio Tras Aplicar Filtro Paso Alto ################

plt.subplot(313)
plotSpec(filteredSignalLow, rate, title_passLow)
plt.colorbar()
plt.tight_layout()


################### Grafico Audio Tras Aplicar Filtro Paso Banda ####################

title_BandPass = "Filtro Paso Banda"
plt.figure(4)
plt.subplot(311)
xFilteredBand,filteredSignalBand = passBandFilter(rate,500,1600,sound)
plotSignalTime(filteredSignalBand,xFilteredBand,title_BandPass)


########## Grafico De La Transformada De Audio Tras Aplicar Filtro Paso Banda ########

xPassBand, fftPassBand = calcFFT(rate, filteredSignalBand)
plt.subplot(312)
plotTransform(xPassBand, fftPassBand,title_BandPass)


############### Espectograma Del Audio Tras Aplicar Filtro Paso Banda ################

plt.subplot(313)
plotSpec(filteredSignalBand, rate, title_BandPass)
plt.colorbar()
plt.tight_layout()

plt.show()

################################# Audio Salida ########################################

scipy.io.wavfile.write('SalidaInversaAlto.wav',rate,filteredSignalHigh)
scipy.io.wavfile.write('SalidaInversaBajo.wav',rate,filteredSignalLow)
scipy.io.wavfile.write('SalidaInversaBanda.wav',rate,filteredSignalBand)







