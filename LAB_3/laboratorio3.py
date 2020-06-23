from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def graphSignal(Titulo,number,signalIn, signalFFT):
    plt.figure(number)
    plt.subplot(121), plt.imshow(signalIn, cmap='gray')
    plt.title(Titulo), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(signalFFT)
    plt.title('FFT'), plt.xticks([]), plt.yticks([])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar()


def calculateFourierTransform(signal):
    f = np.fft.fft2(signal)
    fshift = np.fft.fftshift(f)
    magnitudFFT = 20 * np.log(np.abs(fshift))

    return magnitudFFT

def convolucion(image,kernel):
    #Se calcula el borde
    borde = kernel.shape[0]//2

    ###########Bordes en ceros #############
    #image_convertida=np.zeros(image.shape)
    ########### Mantengo bordes ############
    image_convertida =image.copy()

    for n in range(borde,image.shape[0]-borde):
        for m in range(borde,image.shape[1]-borde):
            valor=0
            for l in range(kernel.shape[0]):
                for k in range(kernel.shape[1]):
                    resultadoKernel= kernel[l][k]*image[n-borde+l][m-borde+k]
                    valor = valor + resultadoKernel
            image_convertida[n][m]= valor
    return image_convertida

####################Se extraen los datos de la imagen##############################

img= plt.imread("leena512.bmp")

###KERNEL SUAVIZADO#######

kernelSuavizado=np.array([[1,4, 6, 4, 1],[4,16, 24, 16, 4],[6,24, 36, 24, 6],[4,16, 24, 16, 4],[1,4, 6, 4, 1]])*(1/256)

####KERNEL DETECTOR DE BORDES #########

kernelDetectorBordes=np.array([[1,2, 0, -2, -1],[1,2, 0, -2, -1],[1,2, 0, -2, -1],[1,2, 0, -2, -1],[1,2, 0, -2, -1]])*(1/256)

############## Aplico convolucion a imagen con los kernel generados #################
print("Realizando cálculos de convolución...")

imageConvSoft = convolucion(img, kernelSuavizado)
imageConvDetector = convolucion(img,kernelDetectorBordes)

print("Cálculos finalizados con éxito.")

############# CALCULO TRANSFORMADA DE FOURIER 2D Y GRAFICAR ##########################

#####Imagen Original ###############

transformImg = calculateFourierTransform(img)
graphSignal('Imagen Entrada',1,img,transformImg)

##### Imagen Suavizada ############

transformImgSoft = calculateFourierTransform(imageConvSoft)
graphSignal('Imagen Con Suavizado',2,imageConvSoft,transformImgSoft)
imagenSoft= Image.fromarray(imageConvSoft.astype('uint8'))
imagenSoft.save("ImagenSuavizada.bmp")


##### Imagen Detector ############

transformImgDetector = calculateFourierTransform(imageConvDetector)
graphSignal('Imagen Con Detección De Bordes',3,imageConvDetector,transformImgDetector)
imagenDetector= Image.fromarray(imageConvDetector.astype('uint8'))
imagenDetector.save("DeteccionBordes.bmp")

##### MOSTRAR GRAFICOS ###########

plt.show()


