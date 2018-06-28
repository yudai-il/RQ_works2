import pandas as pd
import numpy as np
from datetime import datetime,timedelta
# from .optimizer_toolkit import *

import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16008))
from functools import reduce

# date="2017-06-01"

def get_subnew_stocks(stocks,date,subnewThres):
    """
    # 获得某日上市小于N天的次新股
    :param stocks: list 股票列表
    :param date: str eg. "2018-01-01"
    :param N: int 次新股过滤的阈值
    :return: list 列表中的次新股
    """
    return [s for s in stocks if (pd.Timestamp(date) - pd.Timestamp(rqdatac.instruments(s).listed_date)).days<=subnewThres]

def get_st_stocks(stocks,date):
    """
    获得某日的ST类股票
    :param stocks: list 股票列表
    :param date: 交易日
    :return: list 列表中st的股票
    """
    previous_date = rqdatac.get_previous_trading_date(date)
    st_series = rqdatac.is_st_stock(stocks,start_date=previous_date,end_date=date).iloc[-1]
    return st_series[st_series].index.tolist()

def get_suspended_stocks(stocks,date):
    """
    :param stocks:股票列表
    :param date: 检验停牌日期
    :return:
    """
    return [s for s in stocks if rqdatac.is_suspended(s,start_date=date,end_date=date).values[0,0]]

def low_turnover_filter(stocks,date,percentileThres):
    """
    :param stocks: 股票列表
    :param date: 日期
    :param thres: 剔除换手率位于某百分位以下的股票，例如thres=5，则剔除换手率处于5分位以下的股票
    :return:list
    """
    turnover_stks = rqdatac.get_turnover_rate(stocks,date,date,fields="today").iloc[0]
    return turnover_stks[turnover_stks>np.nanpercentile(turnover_stks,q=percentileThres)].index.tolist()

def small_size_filter(stocks,date,percentileThres):
    """
    :param stocks: 股票列表
    :param date: 日期
    :param thres: 剔除市值位于某百分位以下的股票，例如thres=5，则剔除市值处于5分位以下的股票
    :return: list
    """
    marketCap = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.eod_derivative_indicator.a_share_market_val).filter(rqdatac.fundamentals.stockcode.in_(stocks)),entry_date=date)[0,0,:].astype(float)
    return marketCap[marketCap>np.nanpercentile(marketCap,q=percentileThres)].index.tolist()

def noisy_stocks_filter(stocks,date,subnewThres=120,percentileThres=5):
    """
    剔除ST股票、剔除次新股、剔除当日停牌、流动性(市值)某百分位以下的股票
    :param stocks: 股票列表
    :param date: 日期
    :param N: 剔除上市小于N天的次新股
    :param thres: 进行流动性和规模过滤的百分数阈值
    :return: list
    """
    st_stocks = get_st_stocks(stocks,date)
    subnew_stocks = get_subnew_stocks(stocks,date,subnewThres)
    suspended_stks = get_suspended_stocks(stocks,date)
    filtered_stocks = list(set(stocks)-set(st_stocks)-set(subnew_stocks)-set(suspended_stks))
    filtered_stocks = low_turnover_filter(filtered_stocks,date,percentileThres)
    filtered_stocks = small_size_filter(filtered_stocks,date,percentileThres)
    return filtered_stocks


# def classify_beta_marketCap(date,subnewThres,percentileThres):
#     """
#     将全A股根据其size和beta风格暴露度划分成9部分
#     :param date:
#     :return:
#     """
#     latest_trading_date = str(
#         rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))
#     print(latest_trading_date)
#     all_a_stocks = rqdatac.all_instruments(type="CS",date=latest_trading_date).order_book_id.tolist()
#     all_a_stocks = noisy_stocks_filter(all_a_stocks,latest_trading_date,subnewThres,percentileThres)
#     size_marketCap = rqdatac.get_style_factor_exposure(all_a_stocks, latest_trading_date, latest_trading_date, ['size','beta']).sort_index()
#     size_marketCap.index=size_marketCap.index.droplevel(1)
#
#     quantileGroup = size_marketCap.apply(lambda x:pd.cut(x,bins=3,labels=False)+1).reset_index()
#     quantileStocks = quantileGroup.groupby(['size','beta']).apply(lambda x:x.index.tolist())
#     return quantileStocks.apply(lambda x:pd.Series(all_a_stocks).loc[x].values.tolist()).values.tolist()

def calc_factor_returns(date,factors,subnewThres=120,percentileThres=5):
    """
    :param date:日期
    :param factors:因子值
        ['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility',
       'growth', 'book_to_price', 'leverage', 'liquidity', 'non_linear_size']
    :return:
    """

    latest_trading_date = str(
        rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))
    all_a_stocks = rqdatac.all_instruments(type="CS",date=latest_trading_date).order_book_id.tolist()
    all_a_stocks = noisy_stocks_filter(all_a_stocks,latest_trading_date,subnewThres,percentileThres)
    factor_exposure = rqdatac.get_style_factor_exposure(all_a_stocks, latest_trading_date, latest_trading_date, ['size','beta',factors]).sort_index()
    factor_exposure.index=factor_exposure.index.droplevel(1)
    size_marketCap = factor_exposure[['size','beta']]

    quantileGroup = size_marketCap.apply(lambda x:pd.cut(x,bins=3,labels=False)+1).reset_index()
    quantileStocks = quantileGroup.groupby(['size','beta']).apply(lambda x:x.index.tolist())
    market_neutralize_stocks = quantileStocks.apply(
        lambda x: pd.Series(all_a_stocks).loc[x].values.tolist()).values.tolist()

    # factor_exposure = rqdatac.get_style_factor_exposure(all_a_stocks, latest_trading_date, latest_trading_date, factors).sort_index()
    # print(factor_exposure)
    # factor_exposure.index = factor_exposure.index.droplevel(1)
    factor_exposure = factor_exposure[factors]
    def _deuce(series):
        median = series.median()
        return [series[series<=median].index.tolist(),series[series>median].index.tolist()]

    deuceResults = np.array([_deuce(factor_exposure[neutralized_stks]) for neutralized_stks in market_neutralize_stocks]).flatten()

    short_stocksList = list(reduce(lambda x,y:set(x)|set(y),np.array([s for i,s in enumerate(deuceResults) if i%2==0])))
    long_stockList = list(reduce(lambda x,y:set(x)|set(y),np.array([s for i,s in enumerate(deuceResults) if i%2==1])))

    closePrice = rqdatac.get_price_change_rate(all_a_stocks,date,date).iloc[0]

    return closePrice[long_stockList].mean()-closePrice[short_stocksList].mean()








