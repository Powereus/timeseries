import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np;
import tsfunctions as tsfunc;
import scipy.optimize as opt;

# Read CSV file
df=tsfunc.readData();

# Init variables as state one
p11=0.8;
p22=0.8;
mu1=df['gdp'].mean();
mu2= df['gdp'].mean();
sigma1=df['gdp'].std()*0.5;
sigma2=df['gdp'].std()*2;
parameters = [p11,mu1,mu2,sigma1,sigma2]
y=df['gdp'].values;

# Init as average state & Minimise log-likelihood
bnds = ((0, 0.999999999), (-100, 100), (-100, 100), (0, 100), (0, 100));
# Constrained of p1 + p2 = 1 imposed in the LogLikelihoodConstrained function
optimised_params = opt.minimize(fun=tsfunc.LogLikelihoodConstrained, x0=np.asarray(parameters), args=(y,), method='SLSQP', bounds=bnds, tol=1e-10).x

#Convert params to include p22
optimised_params = np.array([optimised_params[0],optimised_params[0],optimised_params[1],optimised_params[2],optimised_params[3],optimised_params[4]]);
#Run the smoother
smoothed_result = tsfunc.HamiltonSmoother(optimised_params,y)

## Plot the result

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