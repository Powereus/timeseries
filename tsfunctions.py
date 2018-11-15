import numpy as np;
import scipy.stats as sp;
import pandas as pd;
from math import log;

def hamilton_filter(p11,p22,mu,sigma,y):
    T = len(y);
    P = [[p11 , 1-p22], [1-p11,p22]];
    predicted_xi = np.zeros(shape=(T+1,2));
    filtered_xi = np.zeros(shape=(T+1,2));
    likelihood = np.zeros(shape=(T+1,2));

    #init values
    predicted_xi[0] = np.array([(1-p22)/(2-p11-p22),(1-p11)/(2-p11-p22)]);

    for i in range(0 , T):
        likelihood[i] = np.array([sp.norm.pdf(y[i],mu[0],sigma[0]),sp.norm.pdf(y[i],mu[1],sigma[1])]);
        filtered_xi[i] = np.array(predicted_xi[i] * likelihood[i] / np.matmul([1,1],predicted_xi[i]*likelihood[i]));
        predicted_xi[i+1]=np.array(P @ filtered_xi[i]);

    return [filtered_xi,predicted_xi]

def LogLikelihood(p11,p22,mu,sigma,y):
    result = hamilton_filter(p11,p22,mu,sigma,y);
    predictedxi = result[1];
    LL = np.log(predictedxi[0:-1,0]*sp.norm.pdf(y,mu[0],sigma[0])+predictedxi[0:-1,1]*sp.norm.pdf(y,mu[1],sigma[1]));
    return np.sum(LL[1:len(y)]) * -1;