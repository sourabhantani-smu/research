#!/usr/bin/env python
# coding: utf-8

import scipy as sp
import matplotlib.pyplot as plt

def plot_matrix(mat, title):
    plt.figure(figsize=(4,4))
    plt.imshow(mat, cmap=plt.cm.gnuplot2.reversed())
    plt.title(title)
    plt.gca().xaxis.tick_bottom()
    plt.show()

def plot_clusters(data, groups, title):
    plt.figure(figsize=(12,12))
    plt.scatter(data[:,0],data[:,1],c=groups, cmap='tab10')
    plt.title(title)
    plt.show()
    plt.close()


def createMatrices(x_list, similarity):
    n = len(x_list)
    W = sp.zeros((n,n)) ## TODO: Is there symmetric/sparse matrix handling built in?
    L = sp.zeros((n,n))
    D = sp.zeros((n,n)) ## TODO: Is diagnoal Matrix handling built in?
    for i in range(n):
        d = 0
        for j in range(i,n):
            if i==j:
                W[i,j] = 0
            else:
                x = similarity(x_list[i],x_list[j])
                W[i,j] = x
                W[j,i] = x
                L[i,j] = -x
                L[j,i] = -x
                d += x
        D[i,i] = d
        L[i,i] = d
    return L,W,D


def convertToEpsilonNeighborhoodGraphByPercentile(oldW, epsilon):
    n=oldW.shape[0]
    weights = []
    for i in range(n):
        for j in range(i,n):
            weights.append(oldW[i,j])
    cutoff = sp.quantile(weights,epsilon)
    print("Cutoff: ",cutoff)
    return convertToEpsilonNeighborhoodGraphByValue(oldW, cutoff)
    
def convertToEpsilonNeighborhoodGraphByValue(oldW, epsilon):
    n=oldW.shape[0]
    W = sp.zeros((n,n)) ## TODO: Is there symmetric/sparse matrix handling built in?
    L = sp.zeros((n,n))
    D = sp.zeros((n,n)) ## TODO: Is diagnoal Matrix handling built in?    
            
    for i in range(n):
        d = 0
        for j in range(n):
            if i==j:
                W[i,j] = 0
            else:
                x = oldW[i,j]
                if(x < epsilon):
                    x = 0
                W[i,j] = x
                W[j,i] = x
                L[i,j] = -x
                L[j,i] = -x
                d += x
        D[i,i] = d
        L[i,i] = d
    return L,W,D,epsilon

    
def plot_histo(x_list):
    plt.figure(figsize=(12,8))
    plt.hist(x_list, bins=100)
    plt.title("Distribution of elements of array")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
