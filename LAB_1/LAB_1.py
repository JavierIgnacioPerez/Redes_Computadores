import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

##################################IMPORTAR SEÑAL DE AUDIO#################################
rate, sound = scipy.io.wavfile.read('handel.wav')

#####################CREAR VECTOR TIEMPO PARA GRAFICAR PUNTOS EN GRAFICA###################
time = np.linspace(0,len(sound)/rate,len(sound))

##########################GRAFICA FUNCION DE AUDIO EN EL TIEMPO############################

plt.figure(1)
plt.title("GRAFICA FUNCIÓN DE AUDIO EN EL TIEMPO\n")
plt.xlabel('Tiempo (S)')
plt.ylabel('Amplitud')
plt.plot(time,sound.real)

###############################APLICACION TRANSFORMADA DE FOURIER############################
fourier = np.fft.fft(sound)
deltaTime = 1/rate
freq = np.fft.fftfreq(len(sound),deltaTime)

#####################GRAFICO DE LA SEÑAL EN DOMINIO DE LA FRECUENCIA#########################

plt.figure(2)
plt.title("GRAFICO DE LA SEÑAL EN DOMINIO DE LA FRECUENCIA\n")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.plot(freq,abs(fourier.real))

##############################CALCULAR TRANSFORMADA INVERSA##################################

inversaFourier = np.fft.ifft(fourier)

#####################GRAFICO DE LA SEÑAL EN DOMINIO DE LA FRECUENCIA#########################

plt.figure(3)
plt.title("GRAFICA FUNCIÓN DE AUDIO EN EL TIEMPO (INVERSA FFT)\n")
plt.xlabel('Tiempo (S)')
plt.ylabel('Amplitud')
plt.plot(time,inversaFourier.real)

##############CALCULO MAXIMO################

maximoPico = np.amax(fourier)
valorDentroPico = maximoPico.real * porcentajeTruncado

##############CREO VECTOR DE CEROS############

vectorResultado = fourier

###############################TRUNCAR TRANSFORMADA#########################################

i=-1
for elemento in np.nditer(fourier):
	i=i+1
	if elemento.real >= valorDentroPico:
		vectorResultado[i] = elemento
	else: vectorResultado[i] = vectorResultado[i] - vectorResultado[i]
		
#####################GRAFICO DE LA SEÑAL EN DOMINIO DE LA FRECUENCIA########################

plt.figure(4)
plt.title("GRAFICO DE LA SEÑAL EN DOMINIO DE LA FRECUENCIA (TRUNCADA)\n")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.plot(freq,abs(vectorResultado.real))

#############################INVERSA TRUNCADO###############################################

inversaTruncado = np.fft.ifft(vectorResultado)

##################GRAFICO DE LA SEÑAL EN DOMINIO DE LA FRECUENCIA INVERSA###################

plt.figure(5)
plt.title("GRAFICA FUNCIÓN DE AUDIO EN EL TIEMPO (INVERSA FFT TRUNCADA)\n")
plt.xlabel('Tiempo (S)')
plt.ylabel('Amplitud')
plt.plot(time,inversaTruncado.real)
plt.show()

############################SONIDO SALIDA ###################################################

inversaTruncado = np.asarray(inversaTruncado.real, dtype=np.int16)
inversaFourier = np.asarray(inversaFourier.real, dtype=np.int16)

scipy.io.wavfile.write('SalidaInversa.wav',rate,inversaFourier)
scipy.io.wavfile.write('SalidaInversaTruncada.wav',rate,inversaTruncado)z



