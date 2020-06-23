#Integrantes : Javier Pérez y Jorge Ayala
#Fecha: 12-07-2019
#Explicación: Programa que permite realizar la modulación digital de una señal.

############################# BLOQUE DE DEFINICIÓN #############################

'''
Funcion que permite generar una señal binaria de largo N.
Entrada: Integer N que representa el largo de la señal binaria.
Salida : Lista de números binarios que representan la señal
'''
def createBinarySignal(n):
    signal = []
    return signal

'''
Funcion que permite modular mediante ASK
Entrada: Entero que representa el bit de la señal a modular
Salida : La portadora de la señal.
'''
def ASKModulation(signal):
    portA = 0
    portB = 0
    if(signal == 1):
        return portA
    if(signal == 0):
        return portB

'''
Funcion que permite modular mediante FSK
Entrada: Entero que representa el bit de la señal a modular
Salida : La portadora de la señal.
'''
def FSKModulation(signal):
    portA = 0
    portB = 0
    if(signal == 1):
        return portA
    if(signal == 0):
        return portB

'''
Funcion que permite modular mediante PSK
Entrada: Entero que representa el bit de la señal a modular
Salida : La portadora de la señal.
'''
def PSKModulation(signal):
    portA = 0
    portB = 0
    if(signal == 1):
        return portA
    if(signal == 0):
        return portB

'''
Función que dado una lista de elementos binarios que representan una señal, realiza la modulación de esta dependiendo de la modulación especificada.
Entrada: Lista de elementos binarios, String que representa el tipo de modulacion.
Salida : Lista con la señal modulada.
'''
def modulateSignal(modulation,signal):
    modulatedSignal = []
    
    if modulation == ASK:
        for bit in signal:
            modulatedSignal.extend(ASKModulation(bit))
    elif modulation == FSK:
        for bit in signal:
            modulatedSignal.extend(FSKModulation(bit))
    elif modulation == PSK:
        for bit in signal:
            modulatedSignal.extend(PSKModulation(bit))
    else:
        print("No ha especificado una modulación correcta.")
        return 0
    return modulatedSignal


############################# BLOQUE PRINCIPAL #############################

print(".....")
