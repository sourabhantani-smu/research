#!/usr/bin/env python
# coding: utf-8

import scipy as sp
import matplotlib.pyplot as plt

def plot_matrix(mat, title):
    plt.figure(figsize=(12,12))
    plt.imshow(mat, cmap=plt.cm.gnuplot2.reversed())
    plt.title(title)
    plt.gca().xaxis.tick_bottom()
    plt.show()

def plot_clusters(data, groups, title):
    plt.figure(figsize=(12,12))
    plt.scatter(range(len(data)),data,c=groups, cmap='tab10')
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
        for j in range(n):
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

def plot_histo(x_list):
    plt.figure(figsize=(12,8))
    plt.hist(x_list, bins=100)
    plt.title("Distribution of elements of array")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()