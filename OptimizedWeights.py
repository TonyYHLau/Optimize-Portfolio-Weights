# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as scs
import scipy.optimize as sco
import tushare as ts
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

def fetch_stock_data(stock_code, stock_name, start, end):
    df = ts.get_hist_data(stock_code, start=start, end=end)
    df = df.close
    df.name = stock_name
    return df


st_code_name ={'000651':'格力电器',
             '600519':'贵州茅台',
             '601318':'中国平安',
             '000858':'五粮液',
             '600887':'伊利股份',
             '000333':'美的集团',
             '601166':'兴业银行',
             '601328':'交通银行',
             '600104':'上汽集团'}

data = pd.DataFrame({'格力电器':fetch_stock_data('000651', '格力电器', '2015-01-01', '2018-10-23')} 
for k , v in st_code_name.items():
    if k == '格力电器' :
        continue
    data = pd.concat([data, pd.DataFram({k:fetch_stock_data(v, k, '2015-01-01', '2018-10-23')})],1)


data = data.dropna()

data.to_excel('stock_data.xlsx')

date = data.pop('date')

newdata = (data/data.iloc[0, :])*100


init_notebook_mode()

st_name=[]
for v in st_code_name.values():
    st_name.append(v)

def trace(df, date, stock):
    return go.Scatter(x = date, y = df[stock], name=stock)


data = [trace(newdata,date,stock) for stock in st_name]
iplot(data)


log_returns = np.log(newdata.pct_change()+1)
log_returns = log_returns.dropna()
log_returns.mean()*252


def normality_test(array):
    print('Norm test p-value %14.3f' % scs.normaltest(array)[1])

for stock in stocks:
    print('\nResults for {}'.format(stock))
    print('-'*32)
    log_data = np.array(log_returns[stock])
    normality_test(log_data)
    
weights = np.random.random(9)
weights /= np.sum(weights)


###################################################################
import matplotlib.pyplot as plt
%matplotlib inline

#generate 1000 portfolios randomly
port_returns = []
port_variance = []
for p in range(1000):
    weights = np.random.random(9)
    weights /=np.sum(weights)
    port_returns.append(np.sum(log_returns.mean()*252*weights))
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights))))

port_returns = np.array(port_returns)
port_variance = np.array(port_variance)


risk_free = 0.03
plt.figure(figsize=(8, 6))
plt.scatter(port_variance, port_returns, c=(port_returns-risk_free)/port_variance, marker = 'o')
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label = 'Sharpe Ratio')

###################################################################    


def stats(weights):
    weights = np.array(weights)
    port_returns = np.sum(log_returns.mean()*weights)*252
    port_variance = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252,weights)))
    return np.array([port_returns, port_variance, port_returns/port_variance])

#minimized Sharpe with nagative values
def min_sharpe(weights):
    return -stats(weights)[2]

#initialize weights
x0 = 9*[1./9]

#bounds between 0 and 1
bnds = tuple((0,1) for x in range(9))

#sum of weights is 1
cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})

#optimize the weight
opts = sco.minimize(min_sharpe, x0, method = 'SLSQP', bounds = bnds, constraints = cons)
opts


optv = sco.minimize(min_variance, 
                    x0, 
                    method = 'SLSQP', 
                    bounds = bnds, 
                    constraints = cons)

def min_variance(weights):
    return statistics(weights)[1]


target_returns = np.linspace(0.0,0.5,50)
target_variance = []
for tar in target_returns:
    cons = ({'type':'eq','fun':lambda x:stats(x)[0]-tar},{'type':'eq','fun':lambda x:np.sum(x)-1})
    res = sco.minimize(min_variance, x0, method = 'SLSQP', bounds = bnds, constraints = cons)
    target_variance.append(res['fun'])

target_variance = np.array(target_variance)

###########################################################################
#plot graph of efficient frontier
plt.figure(figsize = (8,4))

plt.scatter(port_variance, port_returns, c = port_returns/port_variance,marker = 'o')

plt.scatter(target_variance,target_returns, c = target_returns/target_variance, marker = 'x')
#maximum Sharpe
plt.plot(stats(opts['x'])[1], stats(opts['x'])[0], 'r*', markersize = 15.0)

plt.plot(stats(optv['x'])[1], stats(optv['x'])[0], 'y*', markersize = 15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe ratio')
