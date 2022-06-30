
import math
from traceback import print_tb
import numpy as np
#import pandas as pd
import time
import json

start_time = time.time()

#Suma los elementos una matriz (matrix) dentro de la matriz mayor (bigMatrix), 
#desde unos índices iniciales (indexROW, indexCOL).
def sumMatrix(bigMatrix,matrix,indexROW,indexCOL): 
    m=0   
    for i in range(indexROW,indexROW+len(matrix)):
        n=0
        for j in range(indexCOL,indexCOL+len(matrix)):
            bigMatrix[i][j] = bigMatrix[i][j]+ matrix[m][n]
            n = n+1
        m=m+1
  
    return bigMatrix
"""
def showResults(solutions):
    df = pd.DataFrame(solutions)
    df.to_csv("C:/Usuarios/kquispe/Programas/prueba-pyscript.csv")
#-------------------------------------------------------------------
"""
class Resorte():
    def __init__(self, alambre, dext, vtas, altura, luz1, luz2):
        self.alambre = alambre
        self.dext = dext
        self.vtas = vtas
        self.altura = altura
        self.luz1 = luz1
        self.luz2 = luz2

#Input datos resorte (geometrico) Revisar forma de creacion de objetos (constructores)
resorte = Resorte(11,88,10,315,0,0)
#add Sentido

H_helice = resorte.altura-resorte.alambre    
H_extremo1 = resorte.alambre+resorte.luz1
H_extremo2 = resorte.alambre+resorte.luz2
H_cuerpo = H_helice-H_extremo1-H_extremo2

R = (resorte.dext-resorte.alambre)/2
#Input elementos
nodos_x_vta = 24
#Input datos material
youngModulus = 206700 #en MPa
shearModulus = 79500; #en MPa

#Input condiciones de contorno
lownode1 = 2
lownode2 = 11
lownode3 = 20
upnode1 = 174
upnode2 = 182
upnode3 = 190

#Input Desplazamiento para simulacion
deltaY = -20

#Calculos de geometría
area = 0.25*math.pi*math.pow(resorte.alambre,2) #en mm2
inercia = 0.25*math.pi*math.pow(resorte.alambre/2,4) #en mm4
inerciapolar = inercia*2 #en mm4

#Calculos de mallado
elementos = resorte.vtas*nodos_x_vta         #Total de elementos
nodos = elementos+1                          #Total de nodos

#Arrays Generales
 #Arrays de nodos
nodeArray = []
nodeRadii = []
nodeTheta = []
nodeVta = []
for i in range(nodos):
    nodeArray.append(i)
    nodeTheta.append(i*360/nodos_x_vta)
    nodeVta.append(i/nodos_x_vta)

#Calcula coordenada X del nodo. Entrada: Posicion angular en grados sexagesimales.
def Node_coordX(nodeValue):
  x = R*math.cos(nodeValue*math.pi/180)
  return x

#Calcula coordenada Y del nodo. Entrada: Posicion angular en fraccion de vuelta.
def Node_coordY(nodeValue): 
    if (nodeValue<=1):
        y = math.pow((nodeValue*360),2)/(360*360/H_extremo1)
    elif (nodeValue>(resorte.vtas-1)):
        y = math.pow((nodeValue*360-resorte.vtas*360),2)/(360*360/(-H_extremo2))+H_helice
    elif (nodeValue>1 and nodeValue<=(resorte.vtas-1)):
        inc = H_cuerpo/((resorte.vtas-2)*360)*360/nodos_x_vta
        y = H_extremo1 + inc*(nodeValue*nodos_x_vta-nodos_x_vta)
  
    return y

#Calcula coordenada Z del nodo. Entrada: Posicion angular en grados sexagesimales.
def Node_coordZ (nodeValue): 
    z = -R*math.sin(nodeValue*math.pi/180)
    return z

#Coordenadas cartesianas de los nodos  
NodeX = [Node_coordX(i) for i in nodeTheta]
NodeY = [Node_coordY(i) for i in nodeVta]
NodeZ = [Node_coordZ(i) for i in nodeTheta]

#Declarar las dimensiones XYZ de cada elemento viga
ElemX=[]
ElemY=[]
ElemZ=[]
Long=[]

#Declarar vectores unitarios axial(x), transversal(z) y vertical(y) del elemento
unit_xX = [] 
unit_zX = []
unit_yX = []
unit_xY = [] 
unit_zY = []
unit_yY = []
unit_xZ = [] 
unit_zZ = []
unit_yZ = []

#Declarar angulos entre ejes locales (xyz) y globales(XYZ) del elemento
ang_xX = []
ang_zX = [] 
ang_yX = []
ang_xY = []
ang_zY = [] 
ang_yY = []
ang_xZ = []
ang_zZ = [] 
ang_yZ = []
 
#Declarar vectores acumuladores de matrices
vectorKlocal = []
vectorT = []
vectorTprime = []
vectorKGlobal=[]

#OPERACIONES POR ELEMENTO   
for ii in range(nodos):
    if ii !=240:

        #Direccion de los elementos
        ElemX.append(NodeX[ii+1]-NodeX[ii])
        
        ElemY.append(NodeY[ii+1]-NodeY[ii])
        
        ElemZ.append(NodeZ[ii+1]-NodeZ[ii])
        Long.append(math.pow(math.pow(ElemX[ii],2)+math.pow(ElemY[ii],2)+math.pow(ElemZ[ii],2),0.5))

        #Unitario direccion axial (x)
        unit_xX.append(ElemX[ii]/Long[ii])
        unit_xY.append(ElemY[ii]/Long[ii])
        unit_xZ.append(ElemZ[ii]/Long[ii])

        #Unitario direccion transversal (z)
        unit_zX.append(-unit_xZ[ii]/abs(unit_xZ[ii])*math.pow(math.pow(unit_xZ[ii],2)/(math.pow(unit_xZ[ii],2)+math.pow(unit_xX[ii],2)),0.5))
        unit_zY.append(0)
        unit_zZ.append(unit_xX[ii]/abs(unit_xX[ii])*math.pow(math.pow(unit_xX[ii],2)/(math.pow(unit_xZ[ii],2)+math.pow(unit_xX[ii],2)),0.5))
                
        #Unitario direccion vertical (y)
        unit_yX.append(unit_xZ[ii]*unit_zY[ii]-unit_xY[ii]*unit_zZ[ii])
        unit_yY.append(unit_xX[ii]*unit_zZ[ii]-unit_xZ[ii]*unit_zX[ii])
        unit_yZ.append(-(unit_xX[ii]*unit_zY[ii]-unit_xY[ii]*unit_zX[ii]))

        #Angulos ejes locales (xyz) vs ejes globales (XYZ)
        ang_xX.append(math.acos(unit_xX[ii])*180/math.pi)
        ang_xY.append(math.acos(unit_xY[ii])*180/math.pi)
        ang_xZ.append(math.acos(unit_xZ[ii])*180/math.pi)
        ang_zX.append(math.acos(unit_zX[ii])*180/math.pi)
        ang_zY.append(math.acos(unit_zY[ii])*180/math.pi)
        ang_zZ.append(math.acos(unit_zZ[ii])*180/math.pi)
        ang_yX.append(math.acos(unit_yX[ii])*180/math.pi)
        ang_yY.append(math.acos(unit_yY[ii])*180/math.pi)
        ang_yZ.append(math.acos(unit_yZ[ii])*180/math.pi)
            
        #Elementos de la matriz me rpiez

        k1 = youngModulus*area/Long[ii];
        k2 = shearModulus*inerciapolar/Long[ii];
        k3 = 12*youngModulus*inercia/math.pow(Long[ii],3);
        k4 = 6*youngModulus*inercia/math.pow(Long[ii],2);
        k5 = 4*youngModulus*inercia/Long[ii];

        matrizRigLocal = np.zeros((12,12))
      
        
        #Asignacion de los elementos a la matriz
        matrizRigLocal[0][0]= k1
        matrizRigLocal[0][6]= -k1
        matrizRigLocal[1][1]= k3
        matrizRigLocal[1][5]= k4
        matrizRigLocal[1][7]= -k3
        matrizRigLocal[1][11]= k4
        matrizRigLocal[2][2]= k3
        matrizRigLocal[2][4]= -k4
        matrizRigLocal[2][8]= -k3
        matrizRigLocal[2][10]= -k4
        matrizRigLocal[3][3]= k2
        matrizRigLocal[3][9]= -k2
        matrizRigLocal[4][2]= -k4
        matrizRigLocal[4][4]= k5
        matrizRigLocal[4][8]= k4
        matrizRigLocal[4][10]= k5/2

        matrizRigLocal[5][1]= k4
        matrizRigLocal[5][5]= k5
        matrizRigLocal[5][7]= -k4
        matrizRigLocal[5][11]= k5/2

        matrizRigLocal[6][0]= -k1
        matrizRigLocal[6][6]= k1

        matrizRigLocal[7][1]= -k3
        matrizRigLocal[7][5]= -k4
        matrizRigLocal[7][7]= k3
        matrizRigLocal[7][11]= -k4

        matrizRigLocal[8][2]= -k3
        matrizRigLocal[8][4]= k4
        matrizRigLocal[8][8]= k3
        matrizRigLocal[8][10]= k4

        matrizRigLocal[9][3]= -k2
        matrizRigLocal[9][9]= k2

        matrizRigLocal[10][2]= -k4
        matrizRigLocal[10][4]= k5/2
        matrizRigLocal[10][8]= k4
        matrizRigLocal[10][10]= k5

        matrizRigLocal[11][1]= k4
        matrizRigLocal[11][5]= k5/2
        matrizRigLocal[11][7]= -k4
        matrizRigLocal[11][11]= k5
        
        #Matriz de transformacion
        matrizTransCoord = np.zeros((12,12))
        
        for u in range(4):
            matrizTransCoord[0+3*u][0+3*u] = unit_xX[ii]
            matrizTransCoord[0+3*u][1+3*u] = unit_xY[ii]
            matrizTransCoord[0+3*u][2+3*u] = unit_xZ[ii]
            matrizTransCoord[1+3*u][0+3*u] = unit_yX[ii]
            matrizTransCoord[1+3*u][1+3*u] = unit_yY[ii]
            matrizTransCoord[1+3*u][2+3*u] = unit_yZ[ii]
            matrizTransCoord[2+3*u][0+3*u] = unit_zX[ii]
            matrizTransCoord[2+3*u][1+3*u] = unit_zY[ii]
            matrizTransCoord[2+3*u][2+3*u] = unit_zZ[ii]
        
        #Almacenar matriz de rigidez del elemento
        vectorKlocal.append(matrizRigLocal)
        #Almacenar Matriz de transformacion
        vectorT.append(matrizTransCoord)
        vectorTprime.append(np.transpose(matrizTransCoord))
        #Producto
        firstProd = []
        matrizRigGlobal = []
        firstProd.append(np.matmul(np.transpose(matrizTransCoord),matrizRigLocal))
        matrizRigGlobal.append(np.matmul(firstProd,matrizTransCoord))
        #Almacenar
        vectorKGlobal.append(matrizRigGlobal)

    #FIN FOR DE OPERACIONES POR ELEMENTO    


 #Crear la supermatriz de rigidez del solido
superMatrix = np.zeros((nodos*6+18,nodos*6+18)) 
#Numero de filas de la supermatriz de rigidez: #Nodos * Grados de libertad de cada nodo (son 6 en 3D). Se suman 18 filas más para las condic. contorno
 
#Incorporar las matrices de rigidez global de cada elemento a la matriz.
for p in range(len(vectorKGlobal)):
    matrix = vectorKGlobal[p][0][0]
    superMatrix = sumMatrix(superMatrix,matrix,(p)*6,(p)*6)

#Utilización de las condiciones de contorno.

for q in range(3):
#UX, UY, UZ de los nodos de la base
    superMatrix[nodos*6 + q][(lownode1-1)*6+q] = 1
    superMatrix[nodos*6 + q + 3][(lownode2-1)*6+q] = 1
    superMatrix[nodos*6 + q + 6][(lownode3-1)*6+q] = 1

#UX, UY, UZ de los nodos del tope
    superMatrix[nodos*6 + q + 9][(upnode1-1)*6+q] = 1
    superMatrix[nodos*6 + q + 12][(upnode2-1)*6+q] = 1
    superMatrix[nodos*6 + q + 15][(upnode3-1)*6+q] = 1

#FX, FY, FZ de los nodos de la base
    superMatrix[(lownode1-1)*6+q][nodos*6 + q] = -1
    superMatrix[(lownode2-1)*6+q][nodos*6 + q + 3] = -1
    superMatrix[(lownode3-1)*6+q][nodos*6 + q + 6] = -1
    
#FX, FY, FZ de los nodos del tope
    superMatrix[(upnode1-1)*6+q][nodos*6 + q + 9] = -1
    superMatrix[(upnode2-1)*6+q][nodos*6 + q + 12] = -1
    superMatrix[(upnode3-1)*6+q][nodos*6 + q + 15] = -1 
   
#Vector de coeficientes independientes:
coef = []
for pp in range(len(superMatrix)):
    coef.append([0])

coef[len(coef)-8]=[deltaY]
coef[len(coef)-5]=[deltaY]
coef[len(coef)-2]=[deltaY]
 
 #Resolver simulacion}
storeForces = []
storeDispl = []
#showResults(superMatrix)
inverse = np.linalg.inv(superMatrix)
pyscript.write("label2","Matriz de resultados")

print(f"El tiempo que demore fue de {time.time() - start_time} segundos!")
solut = np.dot(inverse, coef)


#Matriz de fuerzas en los nodos de las condiciones de contorno
forceMatrix = []
w=0
for vv in range(6): #Son 6 nodos de las condiciones de contorno. Esta matriz tendra 6 filas. 
    forceVect = []
    for uv in range(3): #Cada fila tendra las fuerzas X,Y,Z de los nodos (3 columnas)
        forceVect.append(solut[w][0])
        w = w+1
    forceMatrix.append(forceVect)

prueba = [{'x': 1, 'y': 30},
        {'x':2, 'y':30},
        {'x':3, 'y':30}]

prueba2 = [0,1,2]
pyscript.write("label1",prueba2)


f = open('new_file.json', 'w') 
json.dump(prueba,f)

