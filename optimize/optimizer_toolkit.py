import pandas as pd
import numpy as np
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16008))
from rqdatac import *

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


def trackingError(x,benchmark,union_stks,date,covMat):
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

def portfolioRisk(x,covMat):
    """
    计算投资组合波动率
    :param x:
    :param covMat:
    :return:
    """
    return np.sqrt(np.dot(np.dot(np.matrix(x),covMat*252),np.matrix(x).T).A1[0])


def industry_customized_constraint(order_book_ids,industryConstraints,date):
    """
    返回针对股票池的行业约束
    :param order_book_ids:
    :param industryConstraints:
    :param date:
    :return:
    """
    constraints = []
    industries_labels = missing_industryLabel_handler(order_book_ids,date)

    if industryConstraints is None:
        return []
    elif "*" in industryConstraints.keys() :
        constrainted_industry = sorted(set(industries_labels))
        industryConstraints = {s: industryConstraints.get("*") for s in constrainted_industry}
    else:
        constrainted_industry = sorted(set(industries_labels)& set(industryConstraints.keys()))

    for industry in constrainted_industry:

        industry_stock_position = industries_labels.index.get_indexer(industries_labels[industries_labels==industry].index)

        lower,upper = industryConstraints.get(industry)[0],industryConstraints.get(industry)[1]
        constraints.append({"type":"ineq","fun":lambda x:sum(x[i] for i in industry_stock_position)-lower})
        constraints.append({"type":"ineq","fun":lambda x:sum(x[i] for i in -industry_stock_position)+upper})

    return constraints


def industry_neutralize_constraint(order_book_ids, date,deviation,industryNeutral, benchmark):
    """
    返回相对基准申万1级行业有偏离上下界的约束条件
    :param order_book_ids: list
    :param date: string
    :param benchmark: string
    :param deviation: tuple
    :return:
    """
    if industryNeutral is None or deviation is None:
        return []

    constraints = []
    benchmark_components = index_weights(benchmark, date)
    industry_labels = missing_industryLabel_handler(list(set(benchmark_components.index.tolist()).union(set(order_book_ids))),date)
    benchmark_industry_label = industry_labels.loc[benchmark_components.index.tolist()]
    benchmark_merged_df = pd.concat([benchmark_components, benchmark_industry_label], axis=1)
    benchmark_industry_allocation = benchmark_merged_df.groupby(['index_name']).sum()

    # 获取投资组合行业配置信息
    portfolio_industry_label = industry_labels.loc[order_book_ids]
    portfolio_industry = portfolio_industry_label.unique().tolist()
    constrainted_industry = sorted(set(benchmark_industry_allocation.index)&set(portfolio_industry))

    if isinstance(industryNeutral,list):
        constrainted_industry = sorted(set(industryNeutral)&set(constrainted_industry))
    elif industryNeutral=="*":
        constrainted_industry = constrainted_industry
    else:
        raise Exception("请输入'*'或者申万一级行业列表")

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

def validateConstraints(order_book_ids,bounds,date,industryConstraints,industryNeutral,styleConstraints,styleNeutral):
    def _constraintsCheck(constraints, neutral):
        if constraints is None or neutral is None:
            pass
        elif ("*" in constraints.keys() and len(neutral) > 0) or ("*" == neutral and len(constraints) > 0):
            raise Exception("自定义约束 和 偏离约束 不能存在重叠部分")
        elif set(constraints) & set(neutral):
            raise Exception("自定义约束 和 偏离约束 不能存在重叠部分")

    def _boundsCheck(bounds):
        bounds = bounds.values()
        lowerCumsum = np.sum(s[0] for s in bounds)
        upperCumsum = np.sum(s[1] for s in bounds)

        cons1 = (False in [s[0]<=s[1] and s[0]>=0 for s in bounds])

        # 假设下限之和>1 或者 上限之和<1 或者某资产的上界小于下界，某一资产的下界小于0
        if len(bounds) > 0 and (lowerCumsum>1 or cons1 ):
            raise Exception("请确认个股或行业上下界的合理性")
    _boundsCheck(bounds)
    _boundsCheck(industryConstraints)
    _constraintsCheck(industryConstraints,industryNeutral)
    _constraintsCheck(styleConstraints,styleNeutral)

    missing_industry = set(industryConstraints)- set(missing_industryLabel_handler(order_book_ids,date))
    if missing_industry:
        print("WARNING order_book_ids 中没有股票属于{}行业, 已忽略其行业约束".format(missing_industry))

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

    benchmark_industry_label = missing_industryLabel_handler(list(benchmark_components.index), date)

    benchmark_merged_df = pd.concat([benchmark_components, benchmark_industry_label], axis=1)

    benchmark_industry_allocation = benchmark_merged_df.groupby(['index_name']).sum()

    # 获取投资组合行业配置信息

    portfolio_industry_label = missing_industryLabel_handler(order_book_ids, date)

    portfolio_industry = list(portfolio_industry_label.unique())

    missing_industry = list(set(benchmark_industry_allocation.index) - set(portfolio_industry))

    # 若投资组合均配置了基准所包含的行业，则不需要进行配齐处理

    if (len(missing_industry) == 0):

        return None, None

    else:

        matching_component = benchmark_merged_df.loc[benchmark_merged_df['index_name'].isin(missing_industry)]

    return matching_component['weight'].sum(), matching_component['weight']

def style_customized_constraint(order_book_ids,styleConstraints,date):
    """
    :param order_book_ids: 股票列表
    :param styleConstraints: 风格约束
    :param date: 日期,交易日
    :return:
    """
    constrainedStyle = ['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility', 'growth',
                        'book_to_price',
                        'leverage', 'liquidity', 'non_linear_size']
    if styleConstraints is None:
        return []

    elif "*" in styleConstraints.keys():

        styleConstraints = {s:styleConstraints.get("*") for s in constrainedStyle}
    else:
        constrainedStyle = sorted(set(styleConstraints.keys())&set(constrainedStyle))
    style_factor_exposure = get_style_factor_exposure(order_book_ids,date,date,'all')

    constraints = []

    for factor in constrainedStyle:
        lower,upper = styleConstraints.get(factor)[0],styleConstraints.get(factor)[1]
        constraints.append({"type":'ineq',"fun":lambda x:x.dot(style_factor_exposure[factor])-lower})
        constraints.append({"type":"ineq","fun":lambda x:-x.dot(style_factor_exposure[factor])+upper})
    return constraints


def style_neutralize_constraint(order_book_ids,date,deviation,factors,benchmark):
    """
    :param order_book_ids:  股票列表
    :param date: 日期,交易日
    :param deviation: 偏离
    :param factors: 因子
    :param benchmark: 基准
    :return:
    """

    constrainedStyle = ['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility', 'growth',
                        'book_to_price',
                        'leverage', 'liquidity', 'non_linear_size']

    if factors is None or deviation is None:
        return []

    benchmark_components = index_weights(benchmark,date)
    union_stocks = sorted(set(order_book_ids).union(set(benchmark_components.index)))

    style_factor_exposure = get_style_factor_exposure(union_stocks,date,date,factors='all')
    style_factor_exposure.index = style_factor_exposure.index.droplevel(1)

    constraints = []

    if "*"== factors:
        constrainted_style = constrainedStyle
    elif isinstance(factors,list):
        constrainted_style = sorted(set(constrainedStyle)&set(factors))
    else:
        raise Exception("请在风格偏离约束中指定 * 或者风格因子列表")

    benchmark_components_data = style_factor_exposure.loc[benchmark_components.index,constrainted_style]
    portfolio_data = style_factor_exposure.loc[order_book_ids,constrainted_style]

    benchmark_style_exposure = benchmark_components.dot(benchmark_components_data)

    portfolio_style_constraints = {style :(benchmark_style_exposure[style]*(1-deviation[0]),benchmark_style_exposure[style]*(1+deviation[1])) for style in benchmark_style_exposure.index}

    for style in constrainted_style:
        constraints.append({"type":"ineq","fun":lambda x: x.dot(portfolio_data[style])-portfolio_style_constraints[style][0]})
        constraints.append({"type":"ineq","fun":lambda x: portfolio_style_constraints[style][1] - x.dot(portfolio_data[style])})

    return constraints


    # order_book_ids, date, industryNeutral = "*", benchmark = '000300.XSHG', deviation = (0.03, 0.05)

# order_book_ids = index_components("000300.XSHG","2016-03-20")
# order_book_ids.extend(['002789.XSHE',"300505.XSHE","601020.XSHG"])
# date="2016-03-20"
# shenwan_instrument_industry(order_book_ids,date)['index_name'].reindex(order_book_ids)

def missing_industryLabel_handler(order_book_ids,date):
    industry = shenwan_instrument_industry(order_book_ids,date=date)['index_name'].reindex(order_book_ids)
    missing_stocks = industry[industry.isnull()].index.tolist()

    if len(missing_stocks):
        min_date = pd.to_datetime([instruments(s).listed_date for s in missing_stocks]).min()
        supplemented_data = {}
        for i in range(1,6,1):
            datePoint = (min_date+np.timedelta64(i*22,"D")).date()
            # if datePoint
            industryLabels = shenwan_instrument_industry(missing_stocks,datePoint)['index_name']
            supplemented_data.update(industryLabels.to_dict())
            missing_stocks = sorted(set(missing_stocks) - set(industryLabels.index))
            if len(missing_stocks) == 0:
                break
        industry.loc[supplemented_data.keys()] = pd.Series(supplemented_data)
    return industry

# missing_industryLabel_handler(stocks,"2018-07-01")





