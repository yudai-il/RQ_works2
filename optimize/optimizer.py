import pandas as pd
import numpy as np
import scipy.stats as st
import sys
sys.path.append("..")
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16008))
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from .optimizer_toolkit import *


def indicator_optimization(indicator_series, date, cov_estimator="shrinkage",
                           riskThreshold=None,window=126,bounds = {},
                           industryConstraints=None,styleConstraints=None,benchmarkOptions={}):
    """
    进行指标序列最大化
    当riskThreshold(trackingErrorThreshold)指定值时进行风险约束

    :param indicator_series: 指标序列. 指标应为浮点型变量(例如个股市盈率/个股预期收益/个股夏普率)
    :param date: 优化日期 如"2018-06-20"

    :param cov_estimator: str 波动率约束
            可选参数：
                -"shrinkage" 默认值; 对收益的协方差矩阵进行收缩矩阵
                -"empirical" 计算个股收益的经验协方差矩阵
                -"multifactor" 因子协方差矩阵
    :param window: int 计算收益率协方差矩阵时的回溯交易日数目 默认为126
    :param riskThreshold: float 可接受的最大投资组合波动率

    :param bounds: dict 个股头寸的上下限
                        1、{'*': (0, 0.03)},则所有个股仓位都在0-3%之间;
                        2、{'601099.XSHG':(0,0.3),'601216.XSHG':(0,0.4)},则601099的仓位在0-30%，601216的仓位在0-40%之间

    :param industryConstraints:dict 行业的权重上下限
                        1、{'*': (0, 0.03)},则所有行业权重都在0-3%之间;
                        2、{'银行':(0,0.3),'房地产':(0,0.4)},则银行行业的权重在0-30%，房地产行业的权重在0-40%之间
    :param styleConstraints:dict 风格约束的上下界
                        1、{"*":(1,3)},则所有的风格暴露度在1-3之间

    :param benchmarkOptions dict,优化器的可选参数，该字典接受如下参数选项
                - benchmark: string 组合跟踪的基准指数, 默认为"000300.XSHG"
                - cov_estimator:跟踪误差的协方差矩阵约束
                        可选参数
                        -"shrinkage" 默认值; 对收益的协方差矩阵进行收缩矩阵
                        -"empirical" 对收益进行经验协方差矩阵计算
                - trackingErrorThreshold: float 可容忍的最大跟踪误差
                - industryNeutral: list 设定投资组合相对基准行业存在偏离的行业
                - industryDeviation tuple 投资组合的行业权重相对基准行业权重容许偏离度的上下界
                - styleNeutral: list 设定投资组合相对基准行业存在偏离的barra风格因子
                - styleDeviation tuple 投资组合的风格因子权重相对基准风格容许偏离度的上下界
                - industryMatching:boolean 是否根据基准成分股进行配齐
    :return: pandas.Series优化后的个股权重 ,新股列表list(optional),停牌股list(optional)

    """

    benchmarkOptions = {} if benchmarkOptions is None else benchmarkOptions
    benchmark = "000300.XSHG" if benchmarkOptions.get("benchmark") is None else benchmarkOptions.get("benchmark")
    cov_estimator_benchmarkOptions = benchmarkOptions.get('cov_estimator')
    trackingErrorThreshold = benchmarkOptions.get("trackingErrorThreshold")
    industryNeutral,industryDeviation = benchmarkOptions.get('industryNeutral'),benchmarkOptions.get('industryDeviation')
    styleNeutral, styleDeviation = benchmarkOptions.get("styleNeutral"),benchmarkOptions.get("styleDeviation")
    industryMatching = benchmarkOptions.get("industryMatching")

    # FIXME 对于偏离约束中的两个变量是否加入判断，还是只要存在一个变量为None此约束失效 ？
    """
    industryCons = (industryNeutral is None)^(industryDeviation is None)
    styleCons = (styleNeutral is None)^(styleDeviation is None)
    
    if industryCons or styleCons:
        raise Exception("请同时指定行业(风格)偏离的列表或偏离上下限")
    """

    # 对股票池进行处理，去除空值、剔除异常值、标准化处理
    original_stks = indicator_series.index.tolist()
    indicator_series = indicator_series[~indicator_series.isnull()]
    indicator_series = winsorized_std(indicator_series)
    indicator_series = (indicator_series - indicator_series.mean()) / indicator_series.std()

    weighted_stocks = indicator_series.index.tolist()
    # FIXME 传入参数的合理性检验 ,加上 行业和风格的约束重叠性检验 done
    validateConstraints(weighted_stocks, bounds,date, industryConstraints,industryNeutral,styleConstraints,styleNeutral)
    # 获得成分股和传入的股票列表的并集

    _index_components = index_components(benchmark, date=date)
    union_stks = sorted(set(_index_components).union(set(weighted_stocks)))
    start_date = pd.Timestamp(date) - np.timedelta64(window*2, "D")

    subnew_stks = get_subnew_stocks(union_stks, date, window)
    suspended_stks = get_suspended_stocks(union_stks, start_date, date, window)
    union_stks = sorted(set(union_stks) - set(subnew_stks) - set(suspended_stks))

    daily_returns = get_price(union_stks, start_date, date, fields="close").pct_change().dropna(how='all').iloc[-window:]
    # covariance matrix for assets returns
    lw = LedoitWolf()
    covMat = lw.fit(daily_returns).covariance_ if cov_estimator == "shrinkage" else daily_returns.cov()
    covMat_option = lw.fit(daily_returns).covariance_ if cov_estimator_benchmarkOptions=="shrinkage" else daily_returns.cov()

    neutralized_industry_constraints = industry_neutralize_constraint(union_stks, date,deviation=industryDeviation, industryNeutral=industryNeutral, benchmark=benchmark)
    customized_industry_constraints = industry_customized_constraint(union_stks, industryConstraints, date)

    neutralized_style_constraints = style_neutralize_constraint(union_stks,date,styleDeviation,styleNeutral,benchmark)
    customized_style_constraints = style_customized_constraint(union_stks,styleConstraints,date)

    if "*" in bounds.keys():
        bounds = tuple([bounds.get("*")]*len(weighted_stocks))
    benchmark_only_stks = set(union_stks) - set(weighted_stocks)
    bnds1 = {s: (0, 0) for s in benchmark_only_stks}
    bounds.update(bnds1)
    bnds = tuple([bounds.get(s) if s in bounds else (0, 1) for s in union_stks])

    constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1}]

    # risk constraints

    if riskThreshold is not None:
        constraints.append({"type":"ineq","fun":lambda x: -portfolioRisk(x,covMat)+riskThreshold})
    if trackingErrorThreshold is not None:
        constraints.append({"type":"ineq","fun":lambda x: -trackingError(x,benchmark,union_stks,date,covMat_option) + trackingErrorThreshold})

    constraints.extend(neutralized_industry_constraints)
    constraints.extend(customized_industry_constraints)
    constraints.extend(neutralized_style_constraints)
    constraints.extend(customized_style_constraints)
    constraints = tuple(constraints)
    def objectiveFunction(x):
        # factor exposure or somethings of the portfolio
        values = x.dot(indicator_series.loc[union_stks].replace(np.nan,0).values)
        values = values
        return values

    # initial weights for optimization
    x0 = np.ones(len(union_stks)) / len(union_stks)
    options = {'disp': True}
    res = minimize(objectiveFunction, x0, bounds=bnds, constraints=constraints, method='SLSQP', options=options)

    optimized_weight = pd.Series(res['x'], index=union_stks).reindex(original_stks)
    if industryMatching:
        # 获得未分配行业的权重与成分股权重
        undistributedWeight,supplementedStocks = benchmark_industry_matching(union_stks, benchmark, date)
        optimized_weight = pd.concat([optimized_weight*(1-undistributedWeight),supplementedStocks])
    # resultsWeights = pd.concat([shenwan_instrument_industry(optimized_weight.index.tolist(),date)['index_name'],optimized_weight],axis=1).groupby("index_name").sum()
    return optimized_weight,subnew_stks, suspended_stks
