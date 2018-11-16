import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np;
import tsfunctions as tsfunc

# Read CSV file
df=tsfunc.readData();

# Init variables as state one
p11=0.8;
p22=0.8;
mu1=df['gdp'].mean();
mu2= df['gdp'].mean();
sigma1=df['gdp'].std()*0.5;
sigma2=df['gdp'].std()*2;
parameters = [p11,p22,mu1,mu2,sigma1,sigma2]
y=df['gdp'].values;

# Init as average state & Minimise log-likelihood
#bnds = ((0, 0.999999999), (0, 0.999999999), (-100, 100), (-100, 100), (0, 100), (0, 100));
#result = tsfunc.OptimiseLL(parameters,y,bnds);
#estimated_params = result.x;

#read results
p11=0.980218;
p22=0.98046389;
mu1=2.92971797;
mu2=3.32611686;
sigma1=1.85345487;
sigma2=4.82347797;

parametersSmoother = [0.980218,0.98046389,2.92971797,3.32611686,1.85345487,4.82347797]
#Run the smoother
smoothed_result = tsfunc.HamiltonSmoother(parametersSmoother,y)

#plot the result
# Plot series 1: GDP growth

#prepare dataframe
df['smoothedstate'] = smoothed_result[:-1,1].tolist();

fig, ax1 = plt.subplots();
fig.suptitle("GDP Growth and Smoothed State");
color="tab:blue";
ax1.set_xlabel('Time [days]');
ax1.set_ylabel('Percentage change [%]',color=color);
ax1.plot(df['gdp'],color=color);
ax1.tick_params(axis='y', labelcolor=color)

ax2=ax1.twinx();
color="tab:red";
ax2.set_ylabel('State 2 probability',color=color);
ax2.set_ylim([-1.20,1.20])
ax2.plot(df['smoothedstate'],color=color);
ax2.tick_params(axis='y', labelcolor=color)

plt.show();

print("q1b done");