import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np;
import tsfunctions as tsfunc


#Read CSV file
df=tsfunc.readData();

#init variables as state one
p11=0.8;
p22=0.8;
mu1=df['gdp'].mean();
mu2= df['gdp'].mean();
sigma1=df['gdp'].std()*0.5;
sigma2=df['gdp'].std()*2;
parameters = [p11,p22,mu1,mu2,sigma1,sigma2]
y=df['gdp'].values;

#start in state 1
initState= np.array([1,0]);
#Minimise log-likelihood
bnds = ((0, 0.999999999), (0, 0.999999999), (-100, 100), (-100, 100), (0, 100), (0, 100));
result1a_first = tsfunc.OptimiseLLWithInit(parameters,y,bnds,initState);

#start in state 2
initState= np.array([1,0]);
#Minimise log-likelihood
bnds = ((0, 0.999999999), (0, 0.999999999), (-100, 100), (-100, 100), (0, 100), (0, 100));
result1a_second = tsfunc.OptimiseLLWithInit(parameters,y,bnds,initState);

#init as average state
#Minimise log-likelihood
bnds = ((0, 0.999999999), (0, 0.999999999), (-100, 100), (-100, 100), (0, 100), (0, 100));
result1a_third = tsfunc.OptimiseLL(parameters,y,bnds);

print("Done ex 1a")