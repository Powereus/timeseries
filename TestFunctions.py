import pandas as pd;
import matplotlib.pyplot as plt
import tsfunctions as tsfunc



#Read CSV file
df=pd.read_csv('data/data.csv',parse_dates=[0],skiprows=[0,1],header=None,names=['date','gdp','price']);
df=df.set_index('date');

# Plot series 1: GDP growth
plt.figure(1)
plt.title("GDP Growth")
plt.ylabel('%')
plot1 = df['gdp'].plot();

# Plot series 2: Annualised inflation
plt.figure(2)
plt.title('Annualised Inflation')
plt.ylabel('%')
plot2 = df['price'].plot();

#init variables
p11=0.8;
p22=0.8;
mu1=df['gdp'].mean();
mu2= df['gdp'].mean();
sigma1=df['gdp'].std()*0.5;
sigma2=df['gdp'].std()*2;
parameters = [p11,p22,mu1,mu2,sigma1,sigma2]

y=df['gdp'].values;

#Run Hamilton Filter - TESTED
result = tsfunc.HamiltonFilter(parameters,y);

#Run Loglikelihood - TESTED
ll = tsfunc.LogLikelihood(parameters,y);

#Minimise log-likelihood - TESTED
bnds = ((0, 0.999999999), (0, 0.999999999), (-100, 100), (-100, 100), (0, None), (0, None));
result = tsfunc.OptimiseLL(parameters,y,bnds);

print(result);

