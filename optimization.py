import pandas as pd
import numpy as np
import scipy.stats as st
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16007))
#rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))

from rqdatac import *
from datetime import datetime,timedelta
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def get_subnew_stocks(stocks,date,N):
    """
    :param stocks: 股票列表, list
    :param date: str eg. "2018-01-01"
    :param N: int 次新股过滤的阈值
    :return: list 列表中的次新股
    """
    return [s for s in stocks if len(get_trading_dates(instruments(s).listed_date,date))<=N]

def get_st_stocks(stocks,date):
    """
    :param stocks: 股票列表, list
    :param date: 交易日
    :return: list 列表中st的股票
    """
    previous_date = get_previous_trading_date(date)
    st_series = is_st_stock(stocks,start_date=previous_date,end_date=date).iloc[-1]
    return st_series[st_series].index.tolist()

def get_suspended_stocks(stocks,start_date,end_date,N):
    """
    :param stocks:股票列表 list
    :param date: 交易日
    :return: list 列表中的停牌股
    """
    return [s for s in stocks if True in is_suspended(s,start_date,end_date).values.flatten()[-N:]]

#TODO: handling industry constraints
def indicator_optimization(indicator_series, date, trackingErrorMinization, bounds={}, cov_shrinkage=True,
                           deviation=0.05, N=126, benchmark="000300.XSHG"):
    """
    :param indicator_series: 传入序列（对于取值为空的股票不做权重配置处理）pandas.Series
    :param date: 优化日期 如"2018-06-20"
    :param bounds: 上下界 如 {'000001.XSHE': (0, 0.3),"600519.XSHG":(0.2,0.3)}  默认为{}
    :param trackingErrorMinization:可容忍的最大跟踪误差 None/float
    :param cov_shrinkage:是否为收缩矩阵 默认为True
    :param deviation:投资组合的行业权重相对基准行业权重的容许偏离度，默认为5%（0.05)；
    :param N:计算股票收益率协方差矩阵时的回溯交易日数目 默认为126
    :param benchmark:最小化跟踪误差的标的，默认为 "000300.XSHG"
    :return: pandas.Series优化后的个股权重 ,新股列表list(optional),停牌股list(optional)
    """

    def _winsorized_std(data, n=3):
        mean, std = data.mean(), data.std()
        return data.clip(mean - std * n, mean + std * n)

    # 获得传入值非空的series
    original_stks = indicator_series.index.tolist()
    indicator_series = indicator_series[~indicator_series.isnull()]
    # 获得指数成分股
    _index_components = index_components(benchmark, date=date)
    # 除去异常值
    indicator_series = _winsorized_std(indicator_series)
    # 标准化
    indicator_series = (indicator_series - indicator_series.mean()) / indicator_series.std()
    # 获得取值非空的股票列表
    weighted_stocks = indicator_series.index.tolist()
    # 获得成分股和传入的股票列表的并集
    union_stks = sorted(set(_index_components).union(set(weighted_stocks)))
    start_date = pd.Timestamp(date) - np.timedelta64(N + 100, "D")
    # 假设有跟踪误差的约束,去除次新股和停牌的
    if trackingErrorMinization is not None:
        subnew_stks = get_subnew_stocks(union_stks, date, N)
        suspended_stks = get_suspended_stocks(union_stks,start_date ,date,N)
        union_stks = sorted(set(union_stks) - set(subnew_stks) - set(suspended_stks))
    else:
        subnew_stks = []
        suspended_stks = []

    # TODO    行业约束加上指数行业的偏离限制，优化结果受限
#     constraints_industry = portfolio_industry_neutralize(union_stks, date, benchmark=benchmark, deviation=deviation)
    constraints_industry = []
    
    daily_returns = get_price(list(union_stks), start_date, date, fields="close").pct_change().dropna(how='all').iloc[
                    -N:]
    # covariance matrix for assets returns
    if not cov_shrinkage:
        # common covariance matrix for daily returns
        covMat = daily_returns.cov()
    else:
        # shrinkage covariance matrix for daily returns
        lw = LedoitWolf()
        covMat = lw.fit(daily_returns).covariance_
    print(np.linalg.cond(covMat))
    def trackingError(x):
        # vector of deviations
        _index_weights = index_weights(benchmark, date=date)
        X = x - _index_weights.reindex(union_stks).replace(np.nan, 0).values

        result = np.sqrt(np.dot(np.dot(np.matrix(X), covMat * 252), np.matrix(X).T).A1[0])
        return result

    # 对于只在成分股出现的股票，其权重=0
    benchmark_only_stks = set(union_stks) - set(weighted_stocks)
    bnds1 = {s: (0, 0) for s in benchmark_only_stks}
    bounds.update(bnds1)
    bnds = tuple([bounds.get(s) if s in bounds else (0, 1) for s in union_stks])

    if trackingErrorMinization is None:
        constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1}]
    else:
        constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1},
                       {"type": "ineq", "fun": lambda x: -np.sqrt(trackingError(x)) + trackingErrorMinization}]

    constraints.extend(constraints_industry)
    constraints = tuple(constraints)

    #     目标函数 最大化
    def objectiveFunction(x):
        # factor exposure or somethings of the portfolio
        values = -np.dot(np.matrix(x), np.matrix(indicator_series.reindex(union_stks).replace(np.nan, 0).values).T).A1[
            0]
        values = values + np.linalg.norm(x)
        return values

    # the initial weights for optimization
    x0 = np.ones(len(union_stks)) / len(union_stks)
#    x0 = [0]*len(union_stks)

    options = {'disp': True}
#     'maxiter': 10000, 'ftol': 1e-06
    res = minimize(objectiveFunction, x0, bounds=bnds, constraints=constraints, method='SLSQP',options=options)
    return pd.Series(res['x'], index=union_stks).reindex(original_stks), subnew_stks, suspended_stks


date=get_previous_trading_date("2018-06-20")
indicator_series = get_factor(index_components("000906.XSHG"),"return_on_equity_diluted",date=date)
randomPos = (np.random.randint(1,len(indicator_series),30))
indicator_series = indicator_series.iloc[randomPos]
indicator_series =indicator_series
trackingErrorMinization = 0.1
# bounds = {'601099.XSHG':(0,0.3),'601216.XSHG':(0,0.4),"600089.XSHG":(0,0.5)}
x = indicator_optimization(indicator_series,date,trackingErrorMinization,N=126,cov_shrinkage=True)
