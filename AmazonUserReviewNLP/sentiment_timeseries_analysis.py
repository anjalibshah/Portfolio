# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

product_list = ['B001KXZ808','B00BGO0Q9O','B001N07KUE','B0074BW614']
#product_list = ['B0074BW614']
pal = sns.dark_palette("skyblue", 3, reverse=True)
sns.set()
for product in product_list:
  ## plot Amazon star distribution
  plt.figure('Stars_'+product)
  df = joblib.load('./df_'+product+'.pkl')
  df = df.dropna()
  fig = plt.gcf()
  ax = plt.gca()
  hist = sns.distplot(df.overall,bins=10,kde=False,rug=False)

  # Labels
  plt.xlabel('Rating',fontsize=20)
  plt.ylabel('Reviews', fontsize=20)
  plt.title('Amazon Star Ratings', fontsize=20)
  # Legend
  plt.xlim(0.5,5.5)
  # Axes
  plt.setp(ax.get_xticklabels(), fontsize=14, family='sans-serif')
  plt.setp(ax.get_yticklabels(), fontsize=18, family='sans-serif')
  plt.tight_layout()
  #plt.show(block=False)
  fig.savefig('StarRating_'+product+'.png')

  # Time series tweet sentiment
  plt.figure('sentiment_'+product)
  df_time = joblib.load('./df_time_'+product+'.pkl')
  df_time.head()
  xD = df_time.resample('D',how='mean')
  xW = df_time.resample('7D',how='mean')
  xM = df_time.resample('M',how='mean')
  fig = plt.gcf()
  ax = plt.gca()

  xD.plot(color=pal[0],lw=3,alpha=.5,ax=ax)
  xW.plot(color=pal[1],lw=3,alpha=.75,ax=ax)
  xM.plot(color=pal[2],lw=1.5,ax=ax)

  # Labels
  plt.xlabel('Date',fontsize=20)
  plt.ylabel('Text Sentiment Score', fontsize=20)
  plt.title('Average User Sentiment', fontsize=20)
  # Legend
  leg = plt.legend(['1 d', '1 wk', '1 mo'], fontsize=12, title='Resampling Rate')
  plt.setp(leg.get_title(),fontsize='15')
  plt.ylim(-1.0,+1.7)
  # Axes
  plt.setp(ax.get_xticklabels(), fontsize=14, family='sans-serif')
  plt.setp(ax.get_yticklabels(), fontsize=18, family='sans-serif')
  plt.tight_layout()
  #plt.show(block=False)
  fig.savefig('sentiment_timeseries_'+product+'.png')

  ## Fit ARIMA model
  plt.figure('ARIMA_'+product)
  fig = plt.gcf()
  ax = plt.gca()
  xD = df_time.resample('D',how='mean',fill_method='ffill',label='sentiment')
  #xD.plot(color=pal[0],lw=3,alpha=.5,ax=ax)
  #begin,end = xD.index[-20],'2014-08-01'
  begin,end = xD.index[-100],'2014-08-01'
  model = sm.tsa.ARIMA(xD,(1,1,0)).fit()
  fit = model.plot_predict(begin,end,ax=ax)
  #fit=sm.tsa.ARIMA(xD,(1,1,0)).fit().plot_predict(begin,end,ax=ax)

  # Labels
  #plt.xlabel('Date',fontsize=20)
  plt.ylabel('Text Sentiment Score', fontsize=20)
  plt.title('Forecasted User Sentiment', fontsize=20)
  # Legend
  leg = plt.legend(['forecast', 'sentiment', '95% CI'], fontsize=12, title='')
  plt.setp(leg.get_title(),fontsize='15')
  plt.ylim(-1.0,+1.7)
  # Axes
  plt.setp(ax.get_xticklabels(), fontsize=14, family='sans-serif')
  plt.setp(ax.get_yticklabels(), fontsize=18, family='sans-serif')
  plt.tight_layout()
  #plt.show(block=False)
  fig.savefig('sentiment_timeseries_ARIMA_'+product+'.png')

  ## plot residuals
  plt.figure('residuals_'+product)
  fig = plt.gcf()
  ax = plt.gca()
  model.resid.plot()
  plt.ylabel('Residuals', fontsize=20)
  plt.title('Residuals', fontsize=20)
  plt.setp(ax.get_xticklabels(), fontsize=14, family='sans-serif')
  plt.setp(ax.get_yticklabels(), fontsize=18, family='sans-serif')
  plt.tight_layout()
