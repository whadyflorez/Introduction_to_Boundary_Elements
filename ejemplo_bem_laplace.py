# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Ejemplo Laplace equation BEM connstant elements
# http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node23.html  normalvectors

import numpy as np


#https://pomax.github.io/bezierinfo/legendre-gauss.html
gw=np.array([0.3607615730481386,0.3607615730481386,0.4679139345726910,0.4679139345726910,0.1713244923791704,0.1713244923791704])
gp=np.array([0.6612093864662645,-0.6612093864662645,-0.2386191860831969,0.2386191860831969,-0.9324695142031521,0.9324695142031521])
ngp=6

nodos=np.array([[0,0],\
       [0.25,0],\
       [0.5,0],\
       [0.75,0],\
          [1,0],\
       [1,0.25],\
        [1,0.5],\
       [1,0.75],\
          [1,1],\
       [0.75,1],\
        [0.5,1],\
      [0.25,1],\
         [0,1],\
      [0,0.75],\
        [0,0.5],\
      [0,0.25]])
    
internos=np.array([[0.25,0.75],\
                  [0.5,0.75],\
                  [0.75,0.75],\
                  [0.25,0.5],\
                  [0.5,0.5],\
                  [0.75,0.75],\
                  [0.25,0.25],\
                  [0.5,0.25],\
                  [0.75,0.25]])    
                  
nodo_central=[]
for i in range(0,15):
  nodo_central.append((nodos[i]+nodos[i+1])/2 )
nodo_central.append((nodos[15]+nodos[0])/2)  
   
    
    
def shape(epsilon):
    fi=np.array([0.5*(1-epsilon),0.5*(1+epsilon)])
    return fi

def dshape(epsilon):
    dfi=np.array([-0.5,0.5])
    return dfi
    
    
def ustar(x,y):
    r=np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    if r !=0:
      u=-(1/(2*np.pi))*np.log(r)
    else:
      u=np.nan
    return u 

def qstar(x,y,n):
    r=np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    if r !=0:
      q=(-1/(2*np.pi*r))*(-np.dot(x-y,n)/r)
    else:
      q=np.nan
    return q 
        

def helem(xsource,element):
  x1=element[0]
  x2=element[1]
  xm=(x1+x2)/2
  if np.linalg.norm(xsource-xm)!=0:
      integral=0
      for i in range(0,ngp):
          y=x1*shape(gp[i])[0]+x2*shape(gp[i])[1]
          dy=x1*dshape(gp[i])[0]+x2*dshape(gp[i])[1]
          n=np.array([dy[1],-dy[0]])/np.linalg.norm(dy)
          ds=np.linalg.norm(dy)
          q=qstar(xsource,y,n)
          integral=integral+gw[i]*q*ds
  else:
      integral=0
  return integral    

def gelem(xsource,element):
  x1=element[0]
  x2=element[1]
  xm=(x1+x2)/2
  l=np.linalg.norm(x2-x1)
  if np.linalg.norm(xsource-xm)!=0:
      integral=0
      for i in range(0,ngp):
          y=x1*shape(gp[i])[0]+x2*shape(gp[i])[1]
          dy=x1*dshape(gp[i])[0]+x2*dshape(gp[i])[1]
          ds=np.linalg.norm(dy)
          u=ustar(xsource,y)
          integral=integral+gw[i]*u*ds
  else:
      integral=(l/(2*np.pi))*(np.log(2/l)+1)
  return integral      
       
#verificar integrales
# element=np.array([[0,1],[1,1]])
# xsource=np.array([0.5,1])
# h=helem(xsource,element) 
# g=gelem(xsource,element)  

H=np.zeros((25,25))       
for i in range(0,16):
    for j in range(0,16):
        if j<15:
          element=[nodos[j,:],nodos[j+1,:]]
        if j==15:
          element=[nodos[15,:],nodos[0,:]]
        H[i,j]=helem(nodo_central[i],element)
#coeficientes diagonal con rigid body criterion
for i in range(0,16):
    H[i,i]=0
    for j in range(0,16):
        if i!=j:
            H[i,i]=H[i,i]-H[i,j]
#internal nodes
for i in range(16,25):
    for j in range(0,16):
        if j<15:
          element=[nodos[j,:],nodos[j+1,:]]
        if j==15:
          element=[nodos[15,:],nodos[0,:]]
        H[i,j]=helem(internos[i-16],element)
#coeficientes diagonal con rigid body criterion
for i in range(16,25):
    H[i,i]=0
    for j in range(0,16):
        H[i,i]=H[i,i]-H[i,j]            
        

G=np.zeros((25,16))       
for i in range(0,16):
    for j in range(0,16):
        if j<15:
          element=[nodos[j,:],nodos[j+1,:]]
        if j==15:
          element=[nodos[15,:],nodos[0,:]]
        G[i,j]=gelem(nodo_central[i],element)        
#internal nodes
for i in range(16,25):
    for j in range(0,16):
        if j<15:
          element=[nodos[j,:],nodos[j+1,:]]
        if j==15:
          element=[nodos[15,:],nodos[0,:]]
        G[i,j]=gelem(internos[i-16],element)    

#assembly
A=np.zeros((25,25))
B=np.zeros(25)
X=np.zeros(25)
A[:,0]=H[:,0]
A[:,1]=H[:,1]
A[:,2]=H[:,2]
A[:,3]=H[:,3]
A[:,4]=H[:,8]
A[:,5]=H[:,9]
A[:,6]=H[:,10]
A[:,7]=H[:,11]
A[:,8]=H[:,16]
A[:,9]=H[:,17]
A[:,10]=H[:,18]
A[:,11]=H[:,19]
A[:,12]=H[:,20]
A[:,13]=H[:,21]
A[:,14]=H[:,22]
A[:,15]=H[:,23]
A[:,16]=H[:,24]
A[:,17]=-G[:,4]
A[:,18]=-G[:,5]
A[:,19]=-G[:,6]
A[:,20]=-G[:,7]
A[:,21]=-G[:,12]
A[:,22]=-G[:,13]
A[:,23]=-G[:,14]
A[:,24]=-G[:,15]

B=B-20*H[:,4]
B=B-20*H[:,5]
B=B-20*H[:,6]
B=B-20*H[:,7]
B=B-100*H[:,12]
B=B-100*H[:,13]
B=B-100*H[:,14]
B=B-100*H[:,15]
B=B+0*G[:,0]
B=B+0*G[:,1]
B=B+0*G[:,2]
B=B+0*G[:,3]
B=B+0*G[:,8]
B=B+0*G[:,9]
B=B+0*G[:,10]
B=B+0*G[:,11]

X=np.linalg.solve(A,B)

#organize the solution
u=np.zeros(25)
q=np.zeros(16)
u[0]=X[0]
u[1]=X[1]
u[2]=X[2]
u[3]=X[3]
u[4]=20
u[5]=20
u[6]=20
u[7]=20
u[8]=X[4]
u[9]=X[5]
u[10]=X[6]
u[11]=X[7]
u[12]=100
u[13]=100
u[14]=100
u[15]=100
u[16]=X[8]
u[17]=X[9]
u[18]=X[10]
u[19]=X[11]
u[20]=X[12]
u[21]=X[13]
u[22]=X[14]
u[23]=X[15]
u[24]=X[16]
q[0]=0
q[1]=0
q[2]=0
q[3]=0
q[4]=X[17]
q[5]=X[18]
q[6]=X[19]
q[7]=X[20]
q[8]=0
q[9]=0
q[10]=0
q[11]=0
q[12]=X[21]
q[13]=X[22]
q[14]=X[23]
q[15]=X[24]











