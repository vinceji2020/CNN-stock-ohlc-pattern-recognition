# -*- coding: utf-8 -*-
"""
Pattern Generator
@author: Vince
"""

import random
import numpy as np
import seaborn as sns
import pandas as pd
import mplfinance as mpf

sns.set(style='white', context='notebook', palette='deep')

def trend(mu,sigma,ticks,bars):
    #mu = -0.005/1000
    #sigma = 0.0001
    s0=random.randint(1,100)
    
    t = np.linspace(1, ticks, ticks)
    B = np.random.standard_normal(size = ticks) 
    B = np.cumsum(B) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*B 
    S = s0*np.exp(X) ### geometric brownian motion ###
    #sns.lineplot(t,S)

    ohlc=pd.DataFrame(columns=['open','high','low','close'])
    for i in range(0,bars):
        a=S[i*int(ticks/bars):(i+1)*int(ticks/bars)-1]   
        temp={'open':a[0],'high':max(a),'low':min(a),'close':a[len(a)-1]}
        ohlc=ohlc.append(pd.DataFrame.from_dict(temp,orient="index").T)
    
    #ohlc=ohlc.reset_index(drop=True)    
    #ohlc.set_index(pd.to_datetime(ohlc.index),inplace=True)
    #mpf.plot(ohlc,type='candle')
    
    ohlc=ohlc.reset_index(drop=True) 
    return ohlc


def single_bar(s0,sigma,body,pn):
    high=s0*(1+sigma)
    low=s0*(1-sigma)
    a1=random.uniform(high,low+body*(high-low))
    a2=random.uniform(low,high-body*(high-low))
    if pn==1:
        open0=min(a1,a2)
        close=max(a1,a2)
    elif pn==-1:
        open0=max(a1,a2)
        close=min(a1,a2)
    else:
        open0=s0
        close=s0
    
    return [open0,high,low,close]


"""
z=np.concatenate((X,S))
"""

def evening_star():
    #create upward trend
    mu=random.uniform(0.005,0.03)/1e3
    sigma=mu*15
    n=random.randint(7,100)
    a=trend(mu,sigma,n*100,7)  
    #a=trend(0.002/1e5,random.random()/1e4,n*100,7)
    
    #create last 3 bars
    s0=random.uniform(a.iloc[a.shape[0]-1,1],a.iloc[a.shape[0]-1,3])
    sig=random.uniform(1,1.5)*max(a.std(axis=1)/a.mean(axis=1)) 
    br=random.uniform(0.8,0.99)
    a=a.append(pd.DataFrame([single_bar(s0,sig,br,1)],columns=['open','high','low','close'])) 
 
    s0=random.uniform(a.iloc[a.shape[0]-1,1],a.iloc[a.shape[0]-1,3])
    sig=random.uniform(0.5,1)*min(a.std(axis=1)/a.mean(axis=1))   
    br=random.uniform(0.05,0.2)
    a=a.append(pd.DataFrame([single_bar(s0,sig,0.2,-1)],columns=['open','high','low','close']))  

    s0=random.uniform(a.iloc[a.shape[0]-1,1],a.iloc[a.shape[0]-1,2])
    sig=random.uniform(0.8,2)*(a.std(axis=1)/a.mean(axis=1)).mean() 
    br=random.uniform(0.8,0.99)
    pct=random.uniform(0.998,0.999)
    a=a.append(pd.DataFrame([single_bar(s0*pct,sig,br,-1)],columns=['open','high','low','close']))  

    a=a.reset_index(drop=True)  
    return a

def morning_star():
    #create downward trend
    mu=-random.uniform(0.005,0.03)/1e3
    sigma=mu*15
    n=random.randint(7,100)
    a=trend(mu,sigma,n*100,7)  
    #a=trend(0.002/1e5,random.random()/1e4,n*100,7)
    
    #create last 3 bars
    s0=random.uniform(a.iloc[a.shape[0]-1,2],a.iloc[a.shape[0]-1,3])
    sig=random.uniform(1,1.5)*max(a.std(axis=1)/a.mean(axis=1)) 
    br=random.uniform(0.8,0.99)
    a=a.append(pd.DataFrame([single_bar(s0,sig,br,-1)],columns=['open','high','low','close'])) 
 
    s0=random.uniform(a.iloc[a.shape[0]-1,2],a.iloc[a.shape[0]-1,3])
    sig=random.uniform(0.5,1)*min(a.std(axis=1)/a.mean(axis=1))   
    br=random.uniform(0.05,0.2)
    a=a.append(pd.DataFrame([single_bar(s0,sig,0.2,1)],columns=['open','high','low','close']))  

    s0=random.uniform(a.iloc[a.shape[0]-1,1],a.iloc[a.shape[0]-1,2])
    sig=random.uniform(0.8,2)*(a.std(axis=1)/a.mean(axis=1)).mean() 
    br=random.uniform(0.8,0.99)
    pct=random.uniform(1.005,1.02)*sig+1
    a=a.append(pd.DataFrame([single_bar(s0*pct,sig,br,1)],columns=['open','high','low','close']))  

    a=a.reset_index(drop=True)  
    return a

def bullish_engulfing():
    #create downward trend
    mu=-random.uniform(0.005,0.03)/1e3
    sigma=mu*15
    n=random.randint(8,100)
    a=trend(mu,sigma,n*100,8)  
    #a=trend(0.002/1e5,random.random()/1e4,n*100,7)
    
    #create last 2 bars
    s0=random.uniform(a.iloc[a.shape[0]-1,2],a.iloc[a.shape[0]-1,3])
    sig=random.uniform(1,1.5)*(a.std(axis=1)/a.mean(axis=1)).mean() 
    br=random.uniform(0.01,0.99)
    a=a.append(pd.DataFrame([single_bar(s0,sig,br,-1)],columns=['open','high','low','close'])) 
 
    s0=random.uniform(a.iloc[a.shape[0]-2,2],a.iloc[a.shape[0]-2,3])
    sig=random.uniform(1.2,2)*sig  
    br=random.uniform(0.7,0.99)
    a=a.append(pd.DataFrame([single_bar(s0,sig,br,1)],columns=['open','high','low','close']))  

    a=a.reset_index(drop=True)  
    return a


def bearish_engulfing():
    #create downward trend
    mu=random.uniform(0.005,0.03)/1e3
    sigma=mu*15
    n=random.randint(8,100)
    a=trend(mu,sigma,n*100,8)  
    #a=trend(0.002/1e5,random.random()/1e4,n*100,7)
    
    #create last 2 bars
    s0=random.uniform(a.iloc[a.shape[0]-1,2],a.iloc[a.shape[0]-1,3])
    sig=random.uniform(1,1.5)*(a.std(axis=1)/a.mean(axis=1)).mean() 
    br=random.uniform(0.01,0.99)
    a=a.append(pd.DataFrame([single_bar(s0,sig,br,1)],columns=['open','high','low','close'])) 
 
    s0=random.uniform(a.iloc[a.shape[0]-2,2],a.iloc[a.shape[0]-2,3])
    sig=random.uniform(1.2,2)*sig  
    br=random.uniform(0.7,0.99)
    a=a.append(pd.DataFrame([single_bar(s0,sig,br,-1)],columns=['open','high','low','close']))  

    a=a.reset_index(drop=True)  
    return a

###################sanity check OHLC plot
#for i in range(10):
#    a=bearish_engulfing()
#    a=morning_star()
#    a=evening_star()
#    a=trend(0.002/1e5,random.random()/1e4,10000,10)
#    a.set_index(pd.to_datetime(a.index),inplace=True)
#    mpf.plot(a,type='candle')

