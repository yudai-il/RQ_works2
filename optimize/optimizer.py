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
def indicator_optimization(indicator_series, date, cov_estimator=True,
                        window=126, benchmark="000300.XSHG",options=None):
    """
    :param indicator_series: 指标序列. 指标应为浮点型变量(例如个股市盈率/个股预期收益/个股夏普率)
    :param date: 优化日期 如"2018-06-20"

    # TODO 风险约束
    :param cov_estimator: str 风险约束
            可选参数：
                -"shrinkage" 默认值; 对收益的协方差矩阵进行收缩矩阵
                -"empirical" 对收益进行普通的协方差计算
                -"multifactor" 因子协方差矩阵
    :param window: int 计算收益率协方差矩阵时的回溯交易日数目 默认为126


    # TODO options
    :param options dict,优化器的可选参数，该字典接受如下参数选项

                - bounds: dict 个股头寸的上下限
                        1、{'*': (0, 0.03)},则所有个股仓位都在0-3%之间;
                        2、{'601099.XSHG':(0,0.3),'601216.XSHG':(0,0.4)},则601099的仓位在0-30%，601216的仓位在0-40%之间

                - industryConstraints:dict 行业的权重上下限
                        1、{'*': (0, 0.03)},则所有行业权重都在0-3%之间;
                        2、{'银行':(0,0.3),'房地产':(0,0.4)},则银行行业的权重在0-30%，房地产行业的权重在0-40%之间

                - benchmark: string 组合跟踪的基准指数, 默认为"000300.XSHG"
                - trackingErrorThreshold: float 可容忍的最大跟踪误差
                - industryNeutral: list 设定投资组合相对基准行业存在偏离的行业
                - industryDeviation tuple 投资组合的行业权重相对基准行业权重容许偏离度的上下界
                - styleNeutral: list 设定投资组合相对基准行业存在偏离的barra风格因子
                - styleDeviation tuple 投资组合的风格因子权重相对基准风格容许偏离度的上下界
                - industryMatching:boolean 是否根据基准成分股进行配齐
    :return: pandas.Series优化后的个股权重 ,新股列表list(optional),停牌股list(optional)

    """

    # 跟踪误差的约束
    trackingErrorThreshold = options.get("trackingErrorThreshold")

    # 自定义的个股或者行业上下界
    bounds = {} if options.get("bounds") is None else options.get("bounds")
    industryConstraints = {} if options.get("industryConstraints") is None else options.get("industryConstraints")

    # 自定义相对基准的各种偏离 TODO 加入判断是否为空
    industryNeutral,industryDeviation=options.get('industryNeutral'),options.get('industryDeviation')
    styleNeutral, styleDeviation = options.get("styleNeutral"),options.get("styleDeviation")

    industryMatching = options.get("industryMatching")

    # 对股票池进行处理，去除空值、剔除异常值、标准化处理
    original_stks = indicator_series.index.tolist()
    indicator_series = indicator_series[~indicator_series.isnull()]
    indicator_series = winsorized_std(indicator_series)
    indicator_series = (indicator_series - indicator_series.mean()) / indicator_series.std()

    weighted_stocks = indicator_series.index.tolist()
    # FIXME 传入参数的合理性检验 1
    validateConstraints(weighted_stocks, bounds, industryConstraints, date)

    # 获得成分股和传入的股票列表的并集
    _index_components = index_components(benchmark, date=date)
    union_stks = sorted(set(_index_components).union(set(weighted_stocks)))
    start_date = pd.Timestamp(date) - np.timedelta64(window*2, "D")

    subnew_stks = []
    suspended_stks = []
    # FIXME 可能不仅仅是跟踪误差约束 0
    if trackingErrorThreshold is not None:
        subnew_stks = get_subnew_stocks(union_stks, date, window)
        suspended_stks = get_suspended_stocks(union_stks, start_date, date, window)
        union_stks = sorted(set(union_stks) - set(subnew_stks) - set(suspended_stks))

    # 行业约束-1/偏离度-2/自定义 (待测)
    if isinstance(industryNeutral,list):
        # 行业约束加上指数行业的偏离限制,可能没有可行解
        constraints_industry=portfolio_industry_neutralize(union_stks, date, industryNeutral=industryNeutral, benchmark=benchmark,
                                      deviation=industryDeviation)
    elif "*" in industryConstraints.keys():
        constraintsIndustries = set(shenwan_instrument_industry(union_stks))
        industryConstraints = {s:industryConstraints.get("*") for s in constraintsIndustries}
        constraints_industry = industry_constraint(union_stks, industryConstraints, date)
    elif len(industryConstraints):
        constraints_industry = industry_constraint(union_stks, industryConstraints, date)
    else:
        constraints_industry = {}

    # 风格约束
    if isinstance(styleNeutral,list):
        # TODO barra 风格偏离约束 0
        pass

    # 个股上下界
    # 自定义个股上下界,对于key中有*的个股权重约束,其所有股票的头寸约束同一
    if "*" in bounds.keys():
        bounds = tuple([bounds.get("*")]*len(weighted_stocks))
    # 对于只在成分股出现的股票，其权重=0
    benchmark_only_stks = set(union_stks) - set(weighted_stocks)
    bnds1 = {s: (0, 0) for s in benchmark_only_stks}
    bounds.update(bnds1)
    bnds = tuple([bounds.get(s) if s in bounds else (0, 1) for s in union_stks])

    daily_returns = get_price(union_stks, start_date, date, fields="close").pct_change().dropna(how='all').iloc[
                    -window:]
    # covariance matrix for assets returns
    lw = LedoitWolf()
    # FIXME 暂时，将增加种类
    covMat = lw.fit(daily_returns).covariance_ if cov_estimator=="shrinkage" else daily_returns.cov()
    print(np.linalg.cond(covMat))

    # 跟踪误差的约束 结合 行业约束
    constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1}] if trackingErrorThreshold is None else [{"type": "eq", "fun": lambda x: sum(x) - 1},{"type": "ineq", "fun": lambda x: -np.sqrt(trackingError(x,benchmark,union_stks,date,covMat)) + trackingErrorThreshold}]
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

    optimized_weight = pd.Series(res['x'], index=union_stks).reindex(original_stks), subnew_stks, suspended_stks
    if industryMatching:
        # 获得未分配行业的权重与成分股权重
        undistributedWeight,supplementedStocks = benchmark_industry_matching(union_stks, benchmark, date)
        optimized_weight = pd.concat([optimized_weight*(1-undistributedWeight),supplementedStocks])
    return optimized_weight
