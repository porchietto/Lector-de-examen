'''
Created on 2 sept. 2019

@author: porchietto
para  sugerencias y contacto:   porchietto@gmail.com
                                +543516769632
'''
Debug_level = 0

import numpy as np
import cv2
from pyzbar.pyzbar import decode
from os import scandir, getcwd
import argparse

parser = argparse.ArgumentParser(description='Corrector automatico de Examenes')
parser.add_argument("--ruta", default='C:\CABO\Serie 1', help="Ruta al direcctirio que contiene la imagenes")
parser.add_argument("--url", default='127.0.0.1/index.php', help="url donde se envia la informacion")

args = parser.parse_args()
ruta = args.ruta
url = args.url

reporte = {}

#############  Function to put vertices in clockwise order ######################

#Cargo la Imagen

############# Lista directorio###############################
def ls(ruta = getcwd()):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

img_archivos = ls(ruta)

def rectify(h):
        ''' this function put vertices of square we got, in clockwise order '''
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)

        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
        
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]

        return hnew

for img_archivo in img_archivos:
    img = cv2.imread(ruta + '\\' + img_archivo,cv2.IMREAD_GRAYSCALE)
    print('Imagen examinada:', img_archivo)
    #print('tamaño imagen:',img.size)
    output = {
        'examen': img_archivo,
             }
    # find the barcodes in the image and decode each of the barcodes
    try :
        decode.__init__() 
        decodedObjects = decode(img)
    except :
        print('error decodificando codigo de barra')
    if decodedObjects.__len__() == 0: 
        print('No se decto codigo')
        continue
    
    for obj in decodedObjects:
        #print('Type : ', obj.type)
        #print('Data : ', obj.data,'\n')
        #print(obj)
        pass
    
    materias = 5
    opciones = 4
    #Dimenciones de la Matris de salida 
    respuestas_int = np.zeros(((opciones+1)*materias,52),dtype='int32')
    respuestas_bool = np.zeros(((opciones+1)*materias,52),dtype='int')
    #Pixeles que ocupa una linea de la cuadricula
    linea = np.array((4,4))
    #Pixeles que ocupa un cuadro sin las lineas
    area = np.array((41,26))
    #Pixeles del contorno
    contorno = 11
    #Filas X (pixeles de cada casilla + piexeles de cada linea)
    #Columnas X (pixeles de cada casilla + piexeles de cada linea)
    imagen = respuestas_int.shape*(area+linea)
    
    img_gauss = cv2.GaussianBlur(img,(9,9),0)
    
    img_thresh = cv2.adaptiveThreshold(img_gauss,255,1,1,11,2)
    
    area_grilla = img.size*materias/17.19
    offset_area = 0.15
    #print('arrea grilla:',area_grilla)
    
    #deteccion por contornos
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        respuestas_bool[:] = False
        if area_grilla*(1+offset_area) > cv2.contourArea(i) > area_grilla*(1-offset_area): 
            #El area del contorno es entre un 10%mayor o menor del area estimada de la hoja
            #print('area contorno:',cv2.contourArea(i))
            peri = cv2.arcLength(i,True)
            #print('Perimetro: ', peri)
            approx = cv2.approxPolyDP(i,0.05*peri,True)
            cv2.drawContours(img,[approx],0,0,10)
            try :
                approx=rectify(approx)
            except:
                print('error en el contorno')
            
            pts1 = np.float32([approx[0]+[contorno,contorno],
                               approx[1]+[linea[1]-contorno,contorno],
                               approx[2]+[linea[1]-contorno,linea[0]-contorno],
                               approx[3]+[contorno,linea[0]-contorno]])
            
            pts2 = np.float32([ [0,0],[imagen[1]-1,0],imagen[::-1]-1,[0,imagen[0]-1] ])
            
            M = cv2.getPerspectiveTransform(pts1,pts2)
            img_warp = cv2.warpPerspective(img,M,tuple(imagen[::-1]))
            
            img_integra = cv2.integral(img_warp)
            
            img_draw = np.array(img_warp)
            for fila in range(respuestas_int.shape[0]):
                for columna in range(respuestas_int.shape[1]): 
                    puntero = (area+linea)*(fila,columna)
                    cv2.rectangle(img_draw,tuple(puntero[::-1]),tuple((puntero+(6,6))[::-1]),128,1)
                    respuestas_int[fila,columna] = (img_integra[tuple(puntero)] 
                                                 + img_integra[tuple(puntero+area)]
                                                 - img_integra[puntero[0]+area[0]][puntero[1]]
                                                 - img_integra[puntero[0]][puntero[1]+area[1]])
            casilla_clara = respuestas_int[:,1:].max()
            casilla_oscura = respuestas_int[:,1:].min()
            if casilla_oscura > 170000:
                print('¿examen no completado?')
                casilla_oscura = 170000

            offset_gris = 1.1
            #print('Casilla mas blanca: ', casilla_clara)
            #print('Casilla mas negra: ', casilla_oscura)
            '''
            Se toma el valor mas osuco y el valor mas claro de una casilla 
            y se calcula el punto medio como valor de umbral.
            Se evito usar la columna 0 por ya estar pintada en la grilla y solo tomar los valores pintados a mano
            '''
            umbral = (casilla_clara - casilla_oscura)/2 + casilla_oscura*offset_gris
            #print('Umbral calculado: ', umbral)
            
            for fila in range(respuestas_int.shape[0]):
                for columna in range(respuestas_int.shape[1]):
                    puntero = (area+linea)*(fila,columna)
                    if respuestas_int[fila, columna] < umbral:
                        #print('fila =',fila,'columna =',columna)
                        #print('Valor', respuestas_int[fila][columna])
                        #print('punto A =',puntero)
                        #print('punto B =',puntero+area)
                        #print('punto C =',puntero[0]+area[0],';',puntero[1])
                        #print('punto D =',puntero[0],';',puntero[1]+area[1])
                        if fila % 5 or (fila % 5 == 0 and 0 <= columna <= 1):
                            '''Me limito a corregir las filas de control y de respuestas'''
                            cv2.rectangle(img_draw,tuple(puntero[::-1]),tuple((puntero+area)[::-1]),0,3)
                            respuestas_bool[fila, columna] = 1
            a = []
            for i in range(materias):
                if respuestas_bool[i*5,0] == True and respuestas_bool[i*5,1] == False:
                    a.append(True)
                else:
                    a.append(False)
            if all(a):
                output = {
                        'examen': img_archivo,
                        'codigo': int(obj.data),
                        'Tipo de Codigo': obj.type,
                        'materias': materias,
                        'opciones': opciones,
                        'respuestas': respuestas_bool.tolist()                    
                    }
                print('grilla detectada para examen: ', output['codigo'])
                if Debug_level > 2:
                    cv2.namedWindow('ventana',cv2.WINDOW_NORMAL)
                    cv2.imshow('ventana',img_draw)
                    cv2.waitKey(0)
            else: 
                print('Grilla incorrecta para examen: ', int(obj.data))
                


#cv2.imwrite('C:\CABO\Serie 1\save.jpg', img_draw)

