import pandas as pd
import numpy as np
import scipy.stats as st
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16007))
from datetime import timedelta
from rqdatac import *
from datetime import datetime,timedelta
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def get_subnew_stocks(stocks,date,N):
    """
    :param stocks: 股票列表, list
    :param date: str eg. "2018-01-01"
    :param N: int 次新股过滤的阈值
    :return:
    """
    return [s for s in stocks if instruments(s).days_from_listed(date)<=N]

def get_st_stocks(stocks,date):
    """
    :param stocks: 股票列表, list
    :param date: 交易日
    :return: list
    """
    previous_date = get_previous_trading_date(date)
    st_series = is_st_stock(stocks,start_date=previous_date,end_date=date).iloc[-1]
    return st_series[st_series].index.tolist()

def get_suspended_stocks(stocks,date):
    previous_date = get_previous_trading_date(date)
    return [s for s in stocks if is_suspended(s,previous_date,date).values.flatten()[-1]]

def calc_tracking_error1(portfolios,date,benchmark,N):
    """
    :param portfolios:投资组合的权重 pandas.Series
            601099.XSHG    0.000184
            000025.XSHE    0.000740
            600600.XSHG    0.001583
            600284.XSHG    0.001869
            600637.XSHG    0.001449

    :param date: 日期 string "2018-01-01"
    :param benchmark: 基准 "000300.XSHG","000905.XSHG"
    :param N: 计算协方差矩阵回溯交易日数目 int 126
    :return: 跟踪误差 float
    """
    _index_weights = index_weights(benchmark,date=date)
    stocks = portfolios.index.tolist()
    all_stks = set(stocks).union(set(_index_weights.index.tolist()))

    subnew_stks = get_subnew_stocks(all_stks, date, N)
    suspended_stks = get_suspended_stocks(all_stks, date)
    all_stks = sorted(set(all_stks) - set(subnew_stks) - set(suspended_stks))

    start_date = pd.Timestamp(date) - np.timedelta64(N+100,"D")

    covMat = get_price(all_stks,start_date,date,fields="close").pct_change().iloc[-N:].dropna(how='all').cov()
    omega = np.matrix(covMat.values)*252

    q_p = np.matrix(portfolios.reindex(all_stks).replace(np.nan,0).values)
    q = np.matrix(_index_weights.reindex(all_stks).replace(np.nan,0).values)
    # the deviations from the index
    w = q_p.A1-q.A1
    return np.sqrt(np.dot(np.dot(w.T,omega),w)).A1[0]

def calc_tracking_error2(portfolios,date,benchmark,N):
    """
    :param portfolios:投资组合的权重 pandas.Series
            601099.XSHG    0.000184
            000025.XSHE    0.000740
            600600.XSHG    0.001583
            600284.XSHG    0.001869
            600637.XSHG    0.001449
    :param date: 日期 string "2018-01-01"
    :param benchmark: 基准 string "000300.XSHG","000905.XSHG"
    :param N: 计算协方差矩阵回溯交易日数目 int 126
    :return: 跟踪误差 float
    """
    stocks = portfolios.index.tolist()
    _index_weights = index_weights(benchmark,date=date)
    all_stks = set(stocks).union(set(_index_weights.index.tolist()))

    subnew_stks = get_subnew_stocks(all_stks, date, N)
    suspended_stks = get_suspended_stocks(all_stks, date)
    all_stks = sorted(set(all_stks) - set(subnew_stks) - set(suspended_stks))

    start_date = pd.Timestamp(date) - np.timedelta64(N+100,"D")
    _price = get_price(all_stks,start_date,date,fields="close").iloc[-N-1:]
    _rets = _price.pct_change().dropna(how='all')
    activeReturns = np.dot(_rets,np.matrix(portfolios.reindex(all_stks).replace(np.nan,0).values).T).A1 - np.dot(_rets,np.matrix(_index_weights.reindex(all_stks).replace(np.nan,0)).T).A1
    return pd.Series(activeReturns).dropna().std()*np.sqrt(252)
    
#-------------------------------TestSuits-------------------------------



n=800
scores = np.random.random(n)
scores = scores/sum(scores)
date="2018-01-01"
benchmark = "000300.XSHG"
N=126
portfolios = pd.Series(scores,index_components("000906.XSHG",date=date))

trackingError1 = calc_tracking_error1(portfolios,date,benchmark,N)
trackingError2 = calc_tracking_error2(portfolios,date,benchmark,N)
