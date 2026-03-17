#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:43:06 2026

@author: whadyimac
"""

import numpy as np

xc=np.array([[1.0,1.0],[2.0,1.0],[2.0,2.0],[1.0,2.0],[1.5,1.5]])

def f(r):
    return 1+r

def b(x):
    y=x[0]**2+x[1]**2
    return y

A=np.zeros((5,5))
B=np.zeros(5)

for i in range(5):
    for j in range(5):
        r=np.linalg.norm(xc[i]-xc[j])
        A[i,j]=f(r)
    B[i]=b(xc[i])
        
 
alpha=np.linalg.solve(A,B)  

def b_aprox(x):
    s=0.0
    for i in range(5):
        r=np.linalg.norm(x-xc[i])
        s+=alpha[i]*f(r)
    return s    
 
print(b_aprox(xc[0]))   
print(b_aprox(xc[1])) 
print(b_aprox([1.5,2]))
