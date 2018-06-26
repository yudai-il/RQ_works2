import pandas as pd
import numpy as np
import scipy.stats as st
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16007))
from rqdatac import *
from datetime import datetime,timedelta
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from.optimizer_toolkit import *


# TODO: handling industry constraints
def indicator_optimization(indicator_series, date, cov_shrinkage=True,
                        N=126, benchmark="000300.XSHG",**kwargs):
    """
    :param indicator_series: 指标序列. 指标应为浮点型变量(例如个股市盈率/个股预期收益/个股夏普率)pandas.Series
    :param date: 优化日期 如"2018-06-20"

    :param cov_shrinkage:boolean 是否对收益的协方差矩阵进行收缩矩阵 默认为True
    :param N: int 计算股票收益率协方差矩阵时的回溯交易日数目 默认为126
    :param benchmark: string 组合跟踪的基准指数，默认为 "000300.XSHG"

    :param bounds: dict 个股头寸的上下限,
            例如{'*': (0, 0.03)},则所有个股仓位都在0-3%之间;
             {'601099.XSHG':(0,0.3),'601216.XSHG':(0,0.4)},则601099的仓位在0-30%，601216的仓位在0-40%之间
    :param industryConstraints:dict 行业的权重上下限

    :param trackingErrorThreshold: float 可容忍的最大跟踪误差
    :param industryNeutral: list 设定投资组合相对基准行业存在偏离的行业
    :param industryDeviation tuple 投资组合的行业权重相对基准行业权重容许偏离度的上下界
    :param styleNeutral: 设定投资组合相对基准行业存在偏离的barra风格因子
    :param styleDeviation tuple 投资组合的风格因子权重相对基准风格容许偏离度的上下界
    :param industryMatching:boolean 是否根据基准成分股进行配齐
    :return: pandas.Series优化后的个股权重 ,新股列表list(optional),停牌股list(optional)

    """

    # 跟踪误差的约束
    trackingErrorThreshold = kwargs.get("trackingErrorThreshold")

    # 自定义的个股或者行业上下界
    bounds = {} if kwargs.get("bounds") is None else kwargs.get("bounds")
    industryConstraints = {} if kwargs.get("industryConstraints") is None else kwargs.get("industryConstraints")

    # 自定义相对基准的各种偏离 TODO 加入判断是否为空
    industryNeutral,industryDeviation=kwargs.get('industryNeutral'),kwargs.get('industryDeviation')
    styleNeutral, styleDeviation = kwargs.get("styleNeutral"),kwargs.get("styleDeviation")

    original_stks = indicator_series.index.tolist()
    indicator_series = indicator_series[~indicator_series.isnull()]
    # 获得指数成分股
    _index_components = index_components(benchmark, date=date)
    # 除去异常值
    indicator_series = winsorized_std(indicator_series)
    # 标准化
    indicator_series = (indicator_series - indicator_series.mean()) / indicator_series.std()
    # 获得取值非空的股票列表
    weighted_stocks = indicator_series.index.tolist()
    # 获得成分股和传入的股票列表的并集
    union_stks = sorted(set(_index_components).union(set(weighted_stocks)))
    start_date = pd.Timestamp(date) - np.timedelta64(N*2, "D")

    subnew_stks = []
    suspended_stks = []
    # 需要进行跟踪误差约束
    if trackingErrorThreshold is not None:
        subnew_stks = get_subnew_stocks(union_stks, date, N)
        suspended_stks = get_suspended_stocks(union_stks, start_date, date, N)
        union_stks = sorted(set(union_stks) - set(subnew_stks) - set(suspended_stks))

    #  偏离度约束
    if isinstance(industryNeutral,list):
        # 行业约束加上指数行业的偏离限制,可能没有可行解 TODO 未分配行业的无效上下界
        constraints_industry=portfolio_industry_neutralize(union_stks, date, industryNeutral=industryNeutral, benchmark=benchmark,
                                      deviation=industryDeviation)
    if isinstance(styleNeutral,list):
        # TODO
        pass

    # 自定义上下界/行业约束
    # 对于key中有*的个股权重约束,其所有股票的头寸约束同一
    if "*" in bounds.keys():
        bounds = tuple([bounds.get("*")]*len(weighted_stocks))
    if "*" in industryConstraints.keys():
        constraintsIndustries = set(shenwan_instrument_industry(union_stks))
        industryConstraints = {s:industryConstraints.get("*") for s in constraintsIndustries}

    if len(industryConstraints):
        constraints_industry = industry_constraint(union_stks, industryConstraints, date)

    daily_returns = get_price(list(union_stks), start_date, date, fields="close").pct_change().dropna(how='all').iloc[
                    -N:]
    # covariance matrix for assets returns
    lw = LedoitWolf()
    covMat = lw.fit(daily_returns).covariance_ if cov_shrinkage else daily_returns.cov()
    print(np.linalg.cond(covMat))
    

    benchmark_only_stks = set(union_stks) - set(weighted_stocks)
    # 对于只在成分股出现的股票，其权重=0
    bnds1 = {s: (0, 0) for s in benchmark_only_stks}
    bounds.update(bnds1)
    bnds = tuple([bounds.get(s) if s in bounds else (0, 1) for s in union_stks])

    # 跟踪误差的约束 结合 行业约束
    constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1}] if trackingErrorThreshold is None else [{"type": "eq", "fun": lambda x: sum(x) - 1},{"type": "ineq", "fun": lambda x: -np.sqrt(trackingErrorConstraints(x,benchmark,union_stks,date,covMat)) + trackingErrorThreshold}]
    constraints.extend(constraints_industry)
    constraints = tuple(constraints)

    #     目标函数 最大化
    def objectiveFunction(x):
        # factor exposure or somethings of the portfolio
        values = x.dot(indicator_series.loc[union_stks].replace(np.nan,0).values)
        values = values + np.linalg.norm(x)
        return values

    # initial weights for optimization
    x0 = np.ones(len(union_stks)) / len(union_stks)
    options = {'disp': True}
    res = minimize(objectiveFunction, x0, bounds=bnds, constraints=constraints, method='SLSQP', options=options)
    return pd.Series(res['x'], index=union_stks).reindex(original_stks), subnew_stks, suspended_stks

# --------------------------------TestSuits---------------------------------

# date=get_previous_trading_date("2018-06-20")
# indicator_series = get_factor(index_components("000906.XSHG"),"return_on_equity_diluted",date=date)
# randomPos = (np.random.randint(1,len(indicator_series),30))
# indicator_series = indicator_series.iloc[randomPos]
# indicator_series =indicator_series
# trackingErrorMinization = 0.1
# # bounds = {'601099.XSHG':(0,0.3),'601216.XSHG':(0,0.4),"600089.XSHG":(0,0.5)}
# x = indicator_optimization(indicator_series,date,trackingErrorMinization,N=126,cov_shrinkage=True)
