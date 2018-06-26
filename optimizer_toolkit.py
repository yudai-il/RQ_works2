import pandas as pd
import numpy as np
import scipy.stats as st
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16007))
#rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))

from rqdatac import *
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def get_subnew_stocks(stocks,date,N):
    """
    # 获得某日上市小于N天的次新股
    :param stocks: list 股票列表
    :param date: str eg. "2018-01-01"
    :param N: int 次新股过滤的阈值
    :return: list 列表中的次新股
    """
    return [s for s in stocks if len(get_trading_dates(instruments(s).listed_date,date))<=N]

def get_st_stocks(stocks,date):
    """
    获得某日的ST类股票
    :param stocks: list 股票列表
    :param date: 交易日
    :return: list 列表中st的股票
    """
    previous_date = get_previous_trading_date(date)
    st_series = is_st_stock(stocks,start_date=previous_date,end_date=date).iloc[-1]
    return st_series[st_series].index.tolist()

def get_suspended_stocks(stocks,start_date,end_date,N):
    """
    获得起始日期内未停牌过的股票列表
    :param stocks: list 股票列表
    :param start_date: 交易日
    :return: list 列表中的停牌股
    """
    return [s for s in stocks if True in is_suspended(s,start_date,end_date).values.flatten()[-N:]]

def winsorized_std(rawData, n=3):
    """
    进行数据的剔除异常值处理
    :param rawData: pandas.Series 未处理的数据
    :param n: 标准化去极值的阈值
    :return: pandas.Series 标准化去极值后的数据
    """
    mean, std = rawData.mean(), rawData.std()
    return rawData.clip(mean - std * n, mean + std * n)


def trackingErrorConstraints(x,benchmark,union_stks,date,covMat):
    """
    跟踪误差约束
    :param x: 权重
    :param benchmark: 基准
    :param union_stks:股票并集
    :param date: 优化日期
    :param covMat: 协方差矩阵
    :return: float
    """
    # vector of deviations
    _index_weights = index_weights(benchmark, date=date)
    X = x - _index_weights.reindex(union_stks).replace(np.nan, 0).values

    result = np.sqrt(np.dot(np.dot(np.matrix(X), covMat * 252), np.matrix(X).T).A1[0])
    return result


def industry_constraint(order_book_ids,industryConstraints,date):
    """
    返回针对股票池的行业约束
    :param order_book_ids:
    :param industryConstraints:
    :param date:
    :return:
    """
    constraints = []
    industries_labels = shenwan_instrument_industry(order_book_ids,date)['index_name']

    missing_industry = set(industryConstraints.keys())- set(industries_labels)

    print("WARNING order_book_ids 中没有股票属于{}行业, 已忽略其行业约束".format(missing_industry))

    for industry in industryConstraints:

        industry_stock_position = industries_labels.index.get_indexer(industries_labels[industries_labels==industry].index)

        down,upper = industryConstraints.get(industry)[0],industryConstraints.get(industry)[1]
        constraints.extend([{"type":"ineq","fun":lambda x:sum(x[i] for i in industry_stock_position)-down},
                            {"type":"ineq","fun":lambda x:sum(x[i] for i in -industry_stock_position)+upper}])

    return constraints


def portfolio_industry_neutralize(order_book_ids, date,industryNeutral="*", benchmark='000300.XSHG', deviation=(0.03,0.05)):
    """
    返回相对基准申万1级行业有偏离上下界的约束条件
    :param order_book_ids: list
    :param date: string
    :param benchmark: string
    :param deviation: tuple
    :return:
    """

    constraints = list()

    # 获取基准行业配置信息

    benchmark_components = index_weights(benchmark, date)

    benchmark_industry_label = shenwan_instrument_industry(list(benchmark_components.index), date)['index_name']

    benchmark_merged_df = pd.concat([benchmark_components, benchmark_industry_label], axis=1)

    benchmark_industry_allocation = benchmark_merged_df.groupby(['index_name']).sum()

    # 获取投资组合行业配置信息

    portfolio_industry_label = shenwan_instrument_industry(order_book_ids, date)['index_name']

    portfolio_industry = list(portfolio_industry_label.unique())

    # missing_industry = list(set(benchmark_industry_allocation.index) - set(portfolio_industry))
    # constrainted_industry = list(set(benchmark_industry_allocation.index) - set(missing_industry))
    # 投资组合可能配置了基准没有配置的行业，因此 portfolio_industry 不一定等于 constrainted_industry

    constrainted_industry = sorted(set(benchmark_industry_allocation.index)&set(portfolio_industry))

    if isinstance(industryNeutral,list):
        constrainted_industry = set(industryNeutral)&set(constrainted_industry)
    elif industryNeutral=="*":
        pass
    else:
        raise Exception("请输入'*'或者申万一级行业列表")

    constraints.append({'type': 'eq', 'fun': lambda x: sum(x) - 1})

    portfolio_industry_constraints = dict(
        (industry, (benchmark_industry_allocation.loc[industry]['weight'] * (1 - deviation[0]),
                    benchmark_industry_allocation.loc[industry]['weight'] * (1 + deviation[1]))) for industry in
        constrainted_industry)
    for industry in constrainted_industry:
        industry_stock_position = portfolio_industry_label.index.get_indexer(
            portfolio_industry_label[portfolio_industry_label == industry].index)

        constraints.append({'type': 'ineq', 'fun': lambda x:
        sum(x[i] for i in industry_stock_position) - portfolio_industry_constraints[industry][0]})
        constraints.append({'type': 'ineq', 'fun': lambda x:
        portfolio_industry_constraints[industry][1] - sum(x[i] for i in industry_stock_position)})

    return constraints

def checkParameterValidity(constraints):
    constraints = constraints.values()
    upperCumSum = np.sum([s[1] for s in constraints])
    downCumsum = np.sum(s[0] for s in constraints)
    cons1 = False in [s[0]<=s[1] for s in constraints]
    if upperCumSum>1 or downCumsum>1 or cons1:
        raise Exception("请确认上下界的合理性")


def benchmark_industry_matching(order_book_ids, benchmark, date):
    """
    返回未配置行业的权重之和/未配置每个行业的权重
    :param order_book_ids:
    :param benchmark:
    :param date:
    :return:
    """

    # 获取基准行业配置信息

    benchmark_components = index_weights(benchmark, date)

    benchmark_industry_label = shenwan_instrument_industry(list(benchmark_components.index), date)['index_name']

    benchmark_merged_df = pd.concat([benchmark_components, benchmark_industry_label], axis=1)

    benchmark_industry_allocation = benchmark_merged_df.groupby(['index_name']).sum()

    # 获取投资组合行业配置信息

    portfolio_industry_label = shenwan_instrument_industry((order_book_ids), date)['index_name']

    portfolio_industry = list(portfolio_industry_label.unique())

    missing_industry = list(set(benchmark_industry_allocation.index) - set(portfolio_industry))

    # 若投资组合均配置了基准所包含的行业，则不需要进行配齐处理

    if (len(missing_industry) == 0):

        return None, None

    else:

        matching_component = benchmark_merged_df.loc[benchmark_merged_df['index_name'].isin(missing_industry)]

    return matching_component['weight'].sum(), matching_component['weight']

