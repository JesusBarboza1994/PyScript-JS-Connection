//ESPACIO PARA INCLUIR REFERENCIAS DE OTRAS HOJAS JS QUE SEAN NECESARIAS PARA EJECUTAR ESTE MAIN().
//import {Node_coordX} from "./NodeCoordinates.js"
//import {Node_coordY} from "./NodeCoordinates.js"
//import {Node_coordZ} from "./NodeCoordinates.js"
//
//import {transpose} from "./MatrixFunctions.js"
//import {multiply} from "./MatrixFunctions.js"
//import {sumMatrix} from "./MatrixFunctions.js"

const { random, randomInt } = require("mathjs");
const mathjs = require("mathjs");

/* DATOS DE ENTRADA */

//Input datos resorte (geometrico) Revisar forma de creacion de objetos (constructores)
var resorte = {};
resorte.alambre = 11;
resorte.dext = 88;
resorte.vtas = 10;
resorte.altura = 310;
resorte.luz1 = 0; //Extremo Base
resorte.luz2 = 0; //Extremo Superior
//Add Sentido

var H_helice = resorte.altura-resorte.alambre;    
var H_extremo1 = resorte.alambre+resorte.luz1;
var H_extremo2 = resorte.alambre+resorte.luz2;
var H_cuerpo = H_helice-H_extremo1-H_extremo2;

var R = (resorte.dext-resorte.alambre)/2;
//Input elementos
 var nodos_x_vta = 24;
//Input datos material
var youngModulus = 206700; //en MPa
var shearModulus = 79500; //en MPa
//Input condiciones de contorno. Se coloca el numero de nodo que se usaran. El numero de nodo puede ir desde 0 hasta nodos_x_vta * vtas totales, inclusive.

var lownode1 = 0;
var lownode2 = 8;
var lownode3 = 16;
var upnode1 = 224;
var upnode2 = 232;
var upnode3 = 240;

//Input Desplazamiento para simulacion
var deltaY;

//Calculos de geometría
var area = 0.25*Math.PI*Math.pow(resorte.alambre,2); //en mm2
var inercia = 0.25*Math.PI*Math.pow(resorte.alambre/2,4); //en mm4
var inerciapolar = inercia*2; //en mm4

//Calculos de mallado
 var elementos = resorte.vtas*nodos_x_vta;         //Total de elementos
 var nodos = elementos+1;                          //Total de nodos

//Arrays Generales
 //Arrays de nodos
   var nodeArray = [];
   var nodeRadii = [];
   var nodeTheta = [];
   var nodeVta = [];
   for (var i=0; i<nodos; i++){
      nodeArray.push(i);
      nodeTheta.push(i*360/nodos_x_vta);
      nodeVta.push(i/nodos_x_vta);
   }
 //Coordenadas cartesianas de los nodos  
   var NodeX = nodeTheta.map(Node_coordX);
   var NodeY = nodeVta.map(Node_coordY);
   var NodeZ = nodeTheta.map(Node_coordZ);

 //Declarar las dimensiones XYZ de cada elemento viga
   var ElemX=[]; 
   var ElemY=[]; 
   var ElemZ=[];
   var Long = [];

 //Declarar vectores unitarios axial(x), transversal(z) y vertical(y) del elemento
   var unit_xX = []; var unit_zX = []; var unit_yX = [];
   var unit_xY = []; var unit_zY = []; var unit_yY = [];
   var unit_xZ = []; var unit_zZ = []; var unit_yZ = [];

 //Declarar angulos entre ejes locales (xyz) y globales(XYZ) del elemento
   var ang_xX = []; var ang_zX = []; var ang_yX = [];
   var ang_xY = []; var ang_zY = []; var ang_yY = [];
   var ang_xZ = []; var ang_zZ = []; var ang_yZ = [];
 
 //Declarar vectores acumuladores de matrices
    var vectorKlocal = [];
    var vectorT = [];
    var vectorTprime = [];
    var vectorKGlobal=[];

 //OPERACIONES POR ELEMENTO   
   for (var ii=1; ii<nodos; ii++){ //El indice ii cuenta los elementos. No existe "elemento 0". Sí existe "nodo 0".
   
     var matrizRigLocal = new Array(12);
     var matrizTransCoord = new Array(12);      

    //Direccion de los elementos
       ElemX[ii] = NodeX[ii]-NodeX[ii-1];
       ElemY[ii] = NodeY[ii]-NodeY[ii-1];
       ElemZ[ii] = NodeZ[ii]-NodeZ[ii-1];
       Long[ii] = Math.pow(Math.pow(ElemX[ii],2)+Math.pow(ElemY[ii],2)+Math.pow(ElemZ[ii],2),0.5);

    //Unitario direccion axial (x)
       unit_xX[ii]=ElemX[ii]/Long[ii];
       unit_xY[ii]=ElemY[ii]/Long[ii];
       unit_xZ[ii]=ElemZ[ii]/Long[ii];

    //Unitario direccion transversal (z)
       unit_zX[ii]=-unit_xZ[ii]/Math.abs(unit_xZ[ii])*Math.pow(Math.pow(unit_xZ[ii],2)/(Math.pow(unit_xZ[ii],2)+Math.pow(unit_xX[ii],2)),0.5);
       unit_zY[ii]=0;
       unit_zZ[ii]=unit_xX[ii]/Math.abs(unit_xX[ii])*Math.pow(Math.pow(unit_xX[ii],2)/(Math.pow(unit_xZ[ii],2)+Math.pow(unit_xX[ii],2)),0.5);
            
    //Unitario direccion vertical (y)
       unit_yX[ii]=unit_xZ[ii]*unit_zY[ii]-unit_xY[ii]*unit_zZ[ii];
       unit_yY[ii]=unit_xX[ii]*unit_zZ[ii]-unit_xZ[ii]*unit_zX[ii];
       unit_yZ[ii]=-(unit_xX[ii]*unit_zY[ii]-unit_xY[ii]*unit_zX[ii]);

    //Angulos ejes locales (xyz) vs ejes globales (XYZ)
       ang_xX[ii]=Math.acos(unit_xX[ii])*180/Math.PI;
       ang_xY[ii]=Math.acos(unit_xY[ii])*180/Math.PI;
       ang_xZ[ii]=Math.acos(unit_xZ[ii])*180/Math.PI;
                            
       ang_zX[ii]=Math.acos(unit_zX[ii])*180/Math.PI;
       ang_zY[ii]=Math.acos(unit_zY[ii])*180/Math.PI;
       ang_zZ[ii]=Math.acos(unit_zZ[ii])*180/Math.PI;

       ang_yX[ii]=Math.acos(unit_yX[ii])*180/Math.PI;
       ang_yY[ii]=Math.acos(unit_yY[ii])*180/Math.PI;
       ang_yZ[ii]=Math.acos(unit_yZ[ii])*180/Math.PI;
              
     //Elementos de la matriz de rigidez

       var k1 = youngModulus*area/Long[ii];
       var k2 = shearModulus*inerciapolar/Long[ii];
       var k3 = 12*youngModulus*inercia/Math.pow(Long[ii],3);
       var k4 = 6*youngModulus*inercia/Math.pow(Long[ii],2);
       var k5 = 4*youngModulus*inercia/Long[ii];

     //Creacion de la matriz vacia 12x12

       for (var x = 0; x<matrizRigLocal.length; x++){
         matrizRigLocal[x]=[0,0,0,0,0,0,0,0,0,0,0,0];
       }

       //Asignacion de los elementos a la matriz
        matrizRigLocal[0][0]= k1;
        matrizRigLocal[0][6]= -k1;

        matrizRigLocal[1][1]= k3;
        matrizRigLocal[1][5]= k4;
        matrizRigLocal[1][7]= -k3;
        matrizRigLocal[1][11]= k4;

        matrizRigLocal[2][2]= k3;
        matrizRigLocal[2][4]= -k4;
        matrizRigLocal[2][8]= -k3;
        matrizRigLocal[2][10]= -k4;

        matrizRigLocal[3][3]= k2;
        matrizRigLocal[3][9]= -k2;

        matrizRigLocal[4][2]= -k4;
        matrizRigLocal[4][4]= k5;
        matrizRigLocal[4][8]= k4;
        matrizRigLocal[4][10]= k5/2;
        
        matrizRigLocal[5][1]= k4;
        matrizRigLocal[5][5]= k5;
        matrizRigLocal[5][7]= -k4;
        matrizRigLocal[5][11]= k5/2;
        
        matrizRigLocal[6][0]= -k1;
        matrizRigLocal[6][6]= k1;
        
        matrizRigLocal[7][1]= -k3;
        matrizRigLocal[7][5]= -k4;
        matrizRigLocal[7][7]= k3;
        matrizRigLocal[7][11]= -k4;
        
        matrizRigLocal[8][2]= -k3;
        matrizRigLocal[8][4]= k4;
        matrizRigLocal[8][8]= k3;
        matrizRigLocal[8][10]= k4;
        
        matrizRigLocal[9][3]= -k2;
        matrizRigLocal[9][9]= k2;
        
        matrizRigLocal[10][2]= -k4;
        matrizRigLocal[10][4]= k5/2;
        matrizRigLocal[10][8]= k4;
        matrizRigLocal[10][10]= k5;
        
        matrizRigLocal[11][1]= k4;
        matrizRigLocal[11][5]= k5/2;
        matrizRigLocal[11][7]= -k4;
        matrizRigLocal[11][11]= k5;

       //Matriz de transformacion

       for (var y = 0; y < matrizTransCoord.length; y++){
         matrizTransCoord[y]=[0,0,0,0,0,0,0,0,0,0,0,0];
       }
       for (var u = 0; u<4; u++){
         matrizTransCoord[0+3*u][0+3*u] = unit_xX[ii]; 
         matrizTransCoord[0+3*u][1+3*u] = unit_xY[ii];
         matrizTransCoord[0+3*u][2+3*u] = unit_xZ[ii];
         matrizTransCoord[1+3*u][0+3*u] = unit_yX[ii];
         matrizTransCoord[1+3*u][1+3*u] = unit_yY[ii];
         matrizTransCoord[1+3*u][2+3*u] = unit_yZ[ii];
         matrizTransCoord[2+3*u][0+3*u] = unit_zX[ii];
         matrizTransCoord[2+3*u][1+3*u] = unit_zY[ii];
         matrizTransCoord[2+3*u][2+3*u] = unit_zZ[ii];
       }
       //Almacenar matriz de rigidez del elemento
         vectorKlocal[ii]=matrizRigLocal;
       //Almacenar Matriz de transformacion
         vectorT[ii]=matrizTransCoord;
         vectorTprime[ii]=transpose(matrizTransCoord);
       //Producto
         var firstProd = multiply(transpose(matrizTransCoord),matrizRigLocal);
         var matrizRigGlobal = multiply(firstProd,matrizTransCoord);
       //Almacenar
         vectorKGlobal[ii]=matrizRigGlobal;
   } //FIN FOR DE OPERACIONES POR ELEMENTO    

 //Crear la supermatriz de rigidez del solido (llena de ceros)
   var superMatrix = new Array(nodos*6+18); //Numero de filas de la supermatriz de rigidez: #Nodos * Grados de libertad de cada nodo (son 6 en 3D). Se suman 18 filas más para las condic. contorno
   for (var z = 0; z < superMatrix.length; z++){
     superMatrix[z]=[];
     for (var zz = 0; zz < superMatrix.length; zz++){
       superMatrix[z][zz]=0;
     }
   }
 
 //Incorporar las matrices de rigidez global de cada elemento a la matriz.
   for (var p = 1; p < vectorKGlobal.length; p++){
     var matrix = vectorKGlobal[p];
     superMatrix = sumMatrix(superMatrix,matrix,(p-1)*6,(p-1)*6);
   }


 //Utilización de las condiciones de contorno.

   for (var q = 0; q<3 ; q++){
    //UX, UY, UZ de los nodos de la base
     superMatrix[nodos*6 + q][(lownode1)*6+q] = 1;
     superMatrix[nodos*6 + q + 3][(lownode2)*6+q] = 1;
     superMatrix[nodos*6 + q + 6][(lownode3)*6+q] = 1;

    //UX, UY, UZ de los nodos del tope
     superMatrix[nodos*6 + q + 9][(upnode1)*6+q] = 1;
     superMatrix[nodos*6 + q + 12][(upnode2)*6+q] = 1;
     superMatrix[nodos*6 + q + 15][(upnode3)*6+q] = 1;

    //FX, FY, FZ de los nodos de la base
     superMatrix[(lownode1)*6+q][nodos*6 + q] = -1;
     superMatrix[(lownode2)*6+q][nodos*6 + q + 3] = -1;
     superMatrix[(lownode3)*6+q][nodos*6 + q + 6] = -1;
     
    //FX, FY, FZ de los nodos del tope
     superMatrix[(upnode1)*6+q][nodos*6 + q + 9] = -1;
     superMatrix[(upnode2)*6+q][nodos*6 + q + 12] = -1;
     superMatrix[(upnode3)*6+q][nodos*6 + q + 15] = -1 

   }
  
 //CONFIGURACION DE LA SIMULACION
 //Se determinan cuantas simulaciones se harán (cuantos desplazamientos), a traves de un for.
 //Dentro del for se calcula el vector solucion "solut", luego se ordena en displaceMatrix y forceMatrix
 //StoreForces es un array que almacena las matrices de fuerza forceMatrix de cada simulación.
 //StoreDispl es un array que almacena las matrices de desplazamiento displaceMatrix de cada simulación
 
 var storeForces = [];
 var storeDispl = [];
 //for (var jj = 25; jj<=150 ; jj=jj+25){ //Se configura la simulacion y el incremento de desplazamiento vertical
  var jj=25;
  deltaY = -jj; 
  var coef = [];
     for (var pp = 0; pp< superMatrix.length; pp++){
       coef.push([0]);
     }
     coef[coef.length-8]=[deltaY];
     coef[coef.length-5]=[deltaY];
     coef[coef.length-2]=[deltaY];
   
   //Resolver simulacion
   var inverse = mathjs.inv(superMatrix); 
   var solut = multiply(inverse,coef);
   
   //Matriz de desplazamientos!
   var displaceMatrix = [];
   var w = 0
    for (var v = 0; v<nodos ; v++){ //La matriz tendrá un numero de filas igual al numero de nodos
       var displaceVect = [];
       for (var uu = 0; uu<6 ; uu++){ //Cada fila tendrá los siguientes elementos en orden: Desplazamientos en X,Y,Z, y Rotaciones en X,Y,Z. (6 columnas en total)
        displaceVect.push(solut[w][0]);
        w = w+1
       }
       displaceMatrix[v] = displaceVect;
      }
   
   //Matriz de fuerzas en los nodos de las condiciones de contorno
   var forceMatrix = [];
   
    for (var vv = 0; vv<6 ; vv++){ //Son 6 nodos de las condiciones de contorno. Esta matriz tendra 6 filas. 
       var forceVect = [];
       for (var uv = 0; uv<3 ; uv++){ // Cada fila tendra las fuerzas X,Y,Z de los nodos (3 columnas)
        forceVect.push(solut[w][0]);
        w = w+1;
       }
       forceMatrix[vv] = forceVect;
      }
   
    storeForces.push(forceMatrix);
    storeDispl.push(displaceMatrix);
   //}

  console.log(forceMatrix);


/* POST-SIMULACIÓN */

//RECÁLCULO DE COORDENADAS X,Y,Z DE CADA NODO.

//Como primer ejemplo, se tomara una de las simulaciones realizadas:
var displaceMatrix1 = storeDispl[0];
var forceMatrix1 = storeForces[0];

var newNodeX = []; 
var newNodeZ = [];
var newNodeY = [];
var newElemX=[]; 
var newElemY=[]; 
var newElemZ=[];
var newLong= [];

var u_sX = []; var u_eX = []; var u_nX = [];
var u_sY = []; var u_eY = []; var u_nY = [];
var u_sZ = []; var u_eZ = []; var u_nZ = [];

newNodeX[0]= NodeX[0]+displaceMatrix1[0][0];
newNodeY[0]= NodeY[0]+displaceMatrix1[0][1];
newNodeZ[0]= NodeZ[0]+displaceMatrix1[0][2];
for (var ii = 0; ii<nodos; ii++){ //Se itera desde ii=1 . ii contarìa los Elementos: Va desde 1 hasta nodos_x_vta*vtas. Para nodos, el contador va desde 0.

  //Se nombran coordenadas X,Y,Z de cada nodo.
  
  //Se calculan las coordenadas para un nodo, tomando los desplazamientos obtenidos en la simulacion.
  if(ii>0){

    newNodeX[ii]= NodeX[ii]+displaceMatrix1[ii][0];
    newNodeY[ii]= NodeY[ii]+displaceMatrix1[ii][1];
    newNodeZ[ii]= NodeZ[ii]+displaceMatrix1[ii][2];
  
    
    
    
    //Se calculan las componentes X,Y,Z del ELEMENTO.
    newElemX[ii] = newNodeX[ii]-newNodeX[ii-1];
    newElemY[ii] = newNodeY[ii]-newNodeY[ii-1];
    newElemZ[ii] = newNodeZ[ii]-newNodeZ[ii-1];
    
    //Obtener las nuevas longitudes de cada ELEMENTO.
    newLong[ii] = Math.pow(Math.pow(newElemX[ii],2)+Math.pow(newElemY[ii],2)+Math.pow(newElemZ[ii],2),0.5);
    
    
    
    //Obtener las direcciones e-n-s en cada nodo
    //Direccion tangencial = direccion axial del elemento DEFORMADO. LAS DEMAS DIRECCIONES SERAN LAS PERPENDICULARES. MISMO PROCEDIMIENTO INICIAL PERO CON LOS DESPLAZAMIENTOS INCLUIDOS.
    
    //CALCULO DE NUEVAS DIRECCIONES DE EJES LOCALES PARA CADA ELEMENTO:  
  
//Declarar vectores unitarios axial(x = s), transversal(z = e) y vertical(y = n) del elemento


 //Unitario direccion axial (x = s)
 u_sX[ii]=newElemX[ii]/newLong[ii]; //u_sx
 u_sY[ii]=newElemY[ii]/newLong[ii]; //u_sy
 u_sZ[ii]=newElemZ[ii]/newLong[ii]; //u_sz
 
 //Unitario direccion transversal (z = e)
 u_eX[ii]=-u_sZ[ii]/Math.abs(u_sZ[ii])*Math.pow(Math.pow(u_sZ[ii],2)/(Math.pow(u_sZ[ii],2)+Math.pow(u_sX[ii],2)),0.5);       //u_ex
 u_eY[ii]=0;                                                                                                                 //u_ey
 u_eZ[ii]=u_sX[ii]/Math.abs(u_sX[ii])*Math.pow(Math.pow(u_sX[ii],2)/(Math.pow(u_sZ[ii],2)+Math.pow(u_sX[ii],2)),0.5);        //u_eZ
 
 //Unitario direccion vertical (y = n)
    u_nX[ii]=u_sZ[ii]*u_eY[ii]-u_sY[ii]*u_eZ[ii];     //u_nx
    u_nY[ii]=u_sX[ii]*u_eZ[ii]-u_sZ[ii]*u_eX[ii];     //u_ny
    u_nZ[ii]=-(u_sX[ii]*u_eY[ii]-u_sY[ii]*u_eX[ii]);  //u_nz
    
  }
  } //Fin del For para cada elemento
  
//console.log(newNodeX);
//console.log(newElemY);
//console.log(newLong);


//EQUILIBRIO DE UNA PARTE DEL RESORTE, APLICANDO CORTE EN NODO ii
var acumvect =[];
  for(ii=0;ii<nodos;ii++){

    
    //Coordenadas XYZ del nodo donde se corta el resorte
  var x_cut = newNodeX[ii];
  var y_cut = newNodeY[ii];
  var z_cut = newNodeZ[ii];
  
  //Fuerzas en x, y, z sobre los 3 nodos de la base, que se usaron para las condiciones de contorno
  var f_ix = forceMatrix1[0][0];    
  var f_iy = forceMatrix1[0][1];    
  var f_iz = forceMatrix1[0][2];    
  
  var f_jx = forceMatrix1[1][0];
  var f_jy = forceMatrix1[1][1];
  var f_jz = forceMatrix1[1][2];

  var f_kx = forceMatrix1[2][0];
  var f_ky = forceMatrix1[2][1];
  var f_kz = forceMatrix1[2][2];
  
  //Coordenadas en x, y, z, de los 3 nodos de la base, que se usaron para las condiciones de contorno.
  var xi = newNodeX[lownode1];
  var xj = newNodeX[lownode2];
  var xk = newNodeX[lownode3];
  
  var yi = newNodeY[lownode1];
  var yj = newNodeY[lownode2];
  var yk = newNodeY[lownode3];

  var zi = newNodeZ[lownode1];
  var zj = newNodeZ[lownode2];
  var zk = newNodeZ[lownode3];
  
  //SUMATORIA DE FUERZAS:

  var Px; var Py; var Pz;
  
  Px = -(f_ix+f_jx+f_kx); //Suma las fuerzas X de los 3 nodos de la base en las que se aplico la condicion de contorno.
    Py = -(f_iy+f_jy+f_ky); //Suma las fuerzas Y de los 3 nodos de la base en las que se aplico la condicion de contorno.
    Pz = -(f_iz+f_jz+f_kz) //Suma las fuerzas Z de los 3 nodos de la base en las que se aplico la condicion de contorno.

    //var P = Math.pow((Math.pow(Px,2) + Math.pow(Py,2) + Math.pow(Pz,2),0.5));
    var vectP = [Px, Py, Pz];  
    
    //console.log(vectP);
    //console.log(NodeZ);
    //console.log(displaceMatrix1);
    //console.log(newNodeZ);
    //console.log(zk);

  //SUMATORIA DE MOMENTOS:
    
  var Mx; var My; var Mz;

  Mx = zi*f_iy + zj*f_jy + zk*f_ky - yi*f_iz - yj*f_jz - yk*f_kz + z_cut*Py - y_cut*Pz; //Suma los momentos X causado por la fuerza P, y las fuerzas de los 3 nodos de la base en las que se aplico la condicion de contorno.
    My = xi*f_iz + xj*f_jz + xk*f_kz - zi*f_ix - zj*f_jx - zk*f_kx + x_cut*Pz - z_cut*Px; //Suma los momentos Y causado por la fuerza P, y las fuerzas de los 3 nodos de la base en las que se aplico la condicion de contorno.
    Mz = yi*f_ix + yj*f_jx + yk*f_kx - xi*f_iy - xj*f_jy - xk*f_ky + y_cut*Px - x_cut*Py; //Suma los momentos Z causado por la fuerza P, y las fuerzas de los 3 nodos de la base en las que se aplico la condicion de contorno.

    //var M = Math.sqrt(Mx*Mx + My*My + Mz*Mz);
    var vectM = [Mx, My, Mz];
    
    //Descomponer P y M en direcciones e,n y s del NODO.
    var dir_s = [u_sX[ii],u_sY[ii],u_sZ[ii]];
    var dir_n = [u_nX[ii],u_nY[ii],u_nZ[ii]];
    var dir_e = [u_eX[ii],u_eY[ii],u_eZ[ii]];
    
    
    //var Ps = mathjs.dot(vectP,dir_s);
    //var Pn = mathjs.dot(vectP,dir_n);
    //var Pe = mathjs.dot(vectP,dir_e);

    var Ps = Px*u_sX[ii]+Py*u_sY[ii]+Pz*u_sZ[ii];
    var Pn = Px*u_nX[ii]+Py*u_nY[ii]+Pz*u_nZ[ii];
    var Pe = Px*u_eX[ii]+Py*u_eY[ii]+Pz*u_eZ[ii];
    
    //var Ms = mathjs.dot(vectM,dir_s);
    //var Mn = mathjs.dot(vectM,dir_n);
    //var Me = mathjs.dot(vectM,dir_e);
    
    var Ms = Mx*u_sX[ii]+My*u_sY[ii]+Mz*u_sZ[ii];
    var Mn = Mx*u_nX[ii]+My*u_nY[ii]+Mz*u_nZ[ii];
    var Me = Mx*u_eX[ii]+My*u_eY[ii]+Mz*u_eZ[ii];
    
    //Esfuerzos:
    var ang = 0; //Diametro interior. 
    var e = resorte.alambre*Math.cos(ang*Math.PI/180);
    var n = resorte.alambre*Math.sin(ang*Math.PI/180);
    var Rdef = Math.pow((Math.pow(z_cut,2)+Math.pow(x_cut,2)),0.5);
    var H = newElemY[ii+1]/newLong[ii+1];   
    var K = 1/(Rdef*Math.pow(H,2));
    var G = 1-K*e;
    
    var tau_e = (Pe/area - Ms/inerciapolar*e)/G;
    var tau_n = (Pn/area + Ms/inerciapolar*n)/G;
    var sigma_s = (Ps/area + Me/inercia*n - Mn/inercia*e)/G

    
  //  console.log(u_sX);
  //  console.log(u_eY);
  //  console.log(vectP);
  //  console.log(vectM);
  //  console.log([Ps,Pn,Pe,Ms,Mn,Me]);
    var esf = [tau_e, tau_n, sigma_s];
    acumvect.push(esf);
  }

  console.log(Long);
  console.log(newLong);    
    
    
//FUNCIONES
    //Comentar lo de abajo cuando funcione lo de separar las hojas
    
    function Node_coordX (nodeValue){ //Calcula coordenada X del nodo. Entrada: Posicion angular en grados sexagesimales.
      var x = R*Math.cos(nodeValue*Math.PI/180);
      return x;
    }

function Node_coordY (nodeValue){ //Calcula coordenada Y del nodo. Entrada: Posicion angular en fraccion de vuelta.
//var nodeValue = 1/24;
  var y;
  if (nodeValue<=1){
    y = Math.pow((nodeValue*360),2)/(360*360/H_extremo1);
  } else if (nodeValue>(resorte.vtas-1)){
    y = Math.pow((nodeValue*360-resorte.vtas*360),2)/(360*360/(-H_extremo2))+H_helice;
  } else if (nodeValue>1 && nodeValue<=(resorte.vtas-1)){
    var inc = H_cuerpo/((resorte.vtas-2)*360)*360/nodos_x_vta;
    y = H_extremo1 + inc*(nodeValue*nodos_x_vta-nodos_x_vta);
  }
  return y;
}

function Node_coordZ (nodeValue){ //Calcula coordenada Z del nodo. Entrada: Posicion angular en grados sexagesimales.
  var z = -R*Math.sin(nodeValue*Math.PI/180);
  return z;
}


function transpose(matrix) { // Devuelve la matriz transpuesta
    return matrix[0].map((col, i) => matrix.map(row => row[i]));
  }


function multiply(a, b) {
    var aNumRows = a.length, aNumCols = a[0].length,
        bNumRows = b.length, bNumCols = b[0].length,
        m = new Array(aNumRows);  // initialize array of rows
    for (var r = 0; r < aNumRows; ++r) {
      m[r] = new Array(bNumCols); // initialize the current row
      for (var c = 0; c < bNumCols; ++c) {
        m[r][c] = 0;             // initialize the current cell
        for (var i = 0; i < aNumCols; ++i) {
          m[r][c] += a[r][i] * b[i][c];
        }
      }
    }
    return m;
}

function sumMatrix(bigMatrix,matrix,indexROW,indexCOL){ // Suma los elementos una matriz (matrix) dentro de la matriz mayor (bigMatrix), desde unos índices iniciales (indexROW, indexCOL).
  
  //var bigMatrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  //var matrix = [[1,2],[3,4]];
  //var indexROW=0;
  //var indexCOL=0;
 
  var m=0;
  for (var i=indexROW; i<indexROW+matrix.length;i++){
  var n=0;
    for(var j=indexCOL; j<indexCOL+matrix.length; j++){
      //bigMatrix[i][j] = bigMatrix[i][j]+ matrix[indexROW-i][indexCOL-j];
      bigMatrix[i][j] = bigMatrix[i][j]+ matrix[m][n];
      n = n+1; 
    }
    m=m+1;
  }
  return bigMatrix;
}
