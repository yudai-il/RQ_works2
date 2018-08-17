import pandas as pd
import numpy as np
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))
from rqdatac import *
import rqdatac
import warnings
from scipy.stats import norm
import scipy.spatial as scsp
from.exception import *

"""
feature branch a
此版为方便调试每个模块的功能，在运行效率上存在不足
整合模块详见 feature branch b
"""

def get_subnew_delisted_assets(order_book_ids,date,N,type="CS"):
    """
    # 获得某日上市小于N天的次新股
    :param stocks: list 股票列表
    :param date: str eg. "2018-01-01"
    :param N: int 次新股过滤的阈值
    :return: list 列表中的次新股
    """

    instruments_fun = fund.instruments if type == "Fund" else instruments
    all_detail_instruments = instruments_fun(order_book_ids)
    subnew_assets = [s for s in all_detail_instruments if len(get_trading_dates(s.listed_date,date))<=N] if isinstance(N,int) else []
    delisted_assets = [s for s in all_detail_instruments if (not s.de_listed_date =="0000-00-00") and pd.Timestamp(s.de_listed_date)<pd.Timestamp(date)]

    return subnew_assets,delisted_assets

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

def get_suspended_stocks(stocks,end_date,N):
    """
    获得起始日期内未停牌过的股票列表
    :param stocks: list 股票列表
    :param start_date: 交易日
    :return: list 列表中的停牌股
    """
    start_date = rqdatac.trading_date_offset(end_date,-N)
    volume = get_price(stocks, start_date, end_date, fields="volume")
    suspended_day = (volume == 0).sum(axis=0)
    return suspended_day[suspended_day>0.5*N].index.tolist()

def assetsListHandler(filter,**kwargs):

    st_filter = filter.get("st_filter")

    benchmark = kwargs.get("benchmark")
    date = kwargs.get("date")
    order_book_ids = kwargs.get("order_book_ids")
    window = filter.get("subnew_filter")

    assetsType = kwargs.get("assetsType")

    if assetsType == "CS":
        benchmark_components = index_components(benchmark, date=date) if benchmark is not None else []
        union_stocks = sorted(set(benchmark_components).union(set(order_book_ids)))
        st_stocks = get_st_stocks(union_stocks,date) if st_filter else []

        subnew_stocks,delisted_stocks = get_subnew_delisted_assets(union_stocks, date, window)
        suspended_stocks = get_suspended_stocks(union_stocks,date,window)
        union_stocks = sorted(set(union_stocks) - set(subnew_stocks) - set(delisted_stocks) - set(st_stocks) - set(suspended_stocks))

        subnew_stocks = list(set(subnew_stocks)&set(order_book_ids))
        delisted_stocks = list(set(delisted_stocks)&set(order_book_ids))
        st_stocks = list(set(st_stocks)&set(order_book_ids))
        suspended_stocks = list(set(suspended_stocks)&set(order_book_ids))

        order_book_ids = sorted(set(union_stocks)&set(order_book_ids))

        return order_book_ids,union_stocks,subnew_stocks,delisted_stocks,suspended_stocks,st_stocks
    else:
        assert assetsType == "Fund"
        subnew_funds,delisted_funds = get_subnew_delisted_assets(order_book_ids, date, window,type="Fund")
        order_book_ids = sorted(set(order_book_ids)-set(subnew_funds)-set(delisted_funds))
        return order_book_ids,order_book_ids,subnew_funds,delisted_funds


def ensure_same_type_instruments(order_book_ids):

    instruments_func = rqdatac.instruments
    instrument = instruments_func(order_book_ids[0])
    if instrument is None:
        instrument = rqdatac.fund.instruments(order_book_ids[0])
        if instrument is not None:
            instruments_func = rqdatac.fund.instruments
        else:
            raise InvalidArgument('unknown instrument: {}'.format(order_book_ids[0]))

    all_instruments_detail = instruments_func(order_book_ids)
    instrumentsTypes = ([instrument.type for instrument in all_instruments_detail])
    cons_all_stocks = instrumentsTypes.count("CS") == len(order_book_ids)
    cons_all_funds = instrumentsTypes.count("PublicFund") == len(order_book_ids)

    if not (cons_all_funds or cons_all_stocks):
        raise InvalidArgument("传入的合约[order_book_ids]应该为统一类型")
    assetsType = "Fund" if cons_all_funds else "CS"
    return assetsType

def trackingError(x,**kwargs):
    covMat,_index_weights = kwargs.get("covMat"),kwargs.get("index_weights")
    X = x - _index_weights
    result = np.sqrt(np.dot(np.dot(X, covMat * 252),X))
    return result

def volatility(x,**kwargs):
    covMat = kwargs.get("covMat")
    return np.dot(np.dot(x,covMat),x)

def mean_variance(x,**kwargs):

    annualized_return = kwargs.get("series")
    risk_aversion_coefficient = kwargs.get("risk_aversion_coefficient")

    if not isinstance(annualized_return,pd.Series):
        raise InvalidArgument("在均值方差优化中请指定 预期收益")
    portfolio_volatility = volatility(x,**kwargs)

    return -x.dot(annualized_return) + np.multiply(risk_aversion_coefficient,portfolio_volatility)

def maximizing_indicator_series(x,**kwargs):
    indicator_series = kwargs.get("series")
    return -x.dot(indicator_series)
def maximizing_return_series(x,**kwargs):
    indicator_series = kwargs.get("series")
    return -x.dot(indicator_series)

def risk_budgeting(x,**kwargs):
    riskMetrics = kwargs.get("riskMetrics")
    if riskMetrics == "volatility":
        return np.sqrt(volatility(x, **kwargs))
    else:
        assert riskMetrics == "tracking_error"
        riskBudgets = kwargs.get("riskBudgets").values
        c_m, _index_weights = kwargs.get("covMat"), kwargs.get("index_weights")
        X = x - _index_weights
        total_tracking_error = trackingError(x,**kwargs)
        contribution_to_active_risk = np.multiply(X,np.dot(c_m,X))/total_tracking_error
        c = np.vstack([contribution_to_active_risk,riskBudgets])
        res = sum(scsp.distance.pdist(c))
        return res

def risk_parity(x,**kwargs):

    c_m = kwargs.get("covMat")

    def risk_parity_with_con_obj_fun(x):
        temp1 = np.multiply(x, np.dot(c_m, x))
        temp2 = temp1[:, None]
        return np.sum(scsp.distance.pdist(temp2, "euclidean"))

    if kwargs.get("with_cons"):
        return risk_parity_with_con_obj_fun(x)
    else:
        c=15
        res = np.dot(x, np.dot(c_m, x)) - c * sum(np.log(x))
        return res


def industry_customized_constraint(order_book_ids,industryConstraints,date):
    """
    返回针对股票池的行业约束
    :param order_book_ids:
    :param industryConstraints:
    :param date:
    :return:
    """
    constraints = []

    if industryConstraints is None or len(industryConstraints) == 0:
        return []
    industries_labels = shenwan_instrument_industry(order_book_ids,date)['index_name']
    shenwan_data = pd.DataFrame(industries_labels)
    shenwan_data['values'] = 1
    shenwan_dummy = shenwan_data.reset_index().pivot(index="index",columns="index_name",values="values").fillna(0)

    if "*" in industryConstraints.keys():
        constrainted_industry = sorted(set(industries_labels))
        industryConstraints = {s: industryConstraints.get("*") for s in constrainted_industry}
    else:
        constrainted_industry = sorted(set(industries_labels)& set(industryConstraints.keys()))

    for industry in constrainted_industry:
        lower,upper = industryConstraints.get(industry)[0],industryConstraints.get(industry)[1]
        constraints.append({"type":"ineq","fun":lambda x:x.dot(shenwan_dummy[industry]) - lower})
        constraints.append({"type":"ineq","fun":lambda x:upper - x.dot(shenwan_dummy[industry])})

    return constraints

def industry_neutralize_constraint(order_book_ids, date,deviation,industryNeutral, benchmark):
    """
    返回相对基准申万1级行业有偏离上下界的约束条件
    :param order_book_ids: list 股票列表
    :param date: string 日期
    :param benchmark: string 基准指数
    :param deviation: float 偏离上下限
    :param industryNeutral: 受约束的行业列表
    :return:
    """
    if industryNeutral is None or deviation is None:
        return []

    constraints = []
    benchmark_components = index_weights(benchmark, date)
    industry_labels = shenwan_instrument_industry(list(set(benchmark_components.index.tolist()).union(set(order_book_ids))),date)['index_name']
    shenwan_data = pd.DataFrame(industry_labels)
    shenwan_data['values'] = 1
    shenwan_dummy = shenwan_data.reset_index().pivot(index="index",columns="index_name",values="values").fillna(0).loc[order_book_ids]

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
        raise InvalidArgument("请输入'*'或者申万一级行业列表")

    portfolio_industry_constraints = dict((industry, (benchmark_industry_allocation.loc[industry]['weight'] * (1 - deviation),
                    benchmark_industry_allocation.loc[industry]['weight'] * (1 + deviation))) for industry in constrainted_industry)

    for industry in constrainted_industry:
        lower,upper = portfolio_industry_constraints[industry][0],portfolio_industry_constraints[industry][1]
        constraints.append({"type":"ineq","fun":lambda x:x.dot(shenwan_dummy[industry])-lower})
        constraints.append({"type":"ineq","fun":lambda x:upper - x.dot(shenwan_dummy[industry])})
    return constraints

def validateConstraints(order_book_ids,bounds,date,industryConstraints,industryNeutral,styleConstraints,styleNeutral):
    def _constraintsCheck(constraints, neutral):
        if constraints is None or neutral is None:
            pass
        elif ("*" in constraints.keys() and len(neutral) > 0) or ("*" == neutral and len(constraints) > 0):
            raise InvalidArgument("自定义约束 和 偏离约束 不能存在重叠部分")
        elif set(constraints) & set(neutral):
            raise InvalidArgument("自定义约束 和 偏离约束 不能存在重叠部分")

    def _boundsCheck(bounds):
        lowerCumsum = np.sum(s[0] for s in bounds.values())

        for _key in bounds:
            lower, upper = bounds.get(_key)
            if lower < 0 or lower > 1:
                raise InvalidArgument(u'OPTIMIZER: 约束 {} 的下限 {} 无效'.format(_key, lower))
            if upper < 0 or upper > 1:
                raise InvalidArgument(u'OPTIMIZER: 约束 {} 的上限 {} 无效'.format(_key, upper))
            if lower > upper:
                raise InvalidArgument(u'OPTIMIZER: 约束的下限 {} 高于上限 {}'.format(lower, upper))

        if len(bounds) > 0 and lowerCumsum>1 :
            raise InvalidArgument("OPTIMIZER: 约束的下限之和大于1")

    _boundsCheck(bounds)
    _boundsCheck(industryConstraints)


    _constraintsCheck(industryConstraints,industryNeutral)
    _constraintsCheck(styleConstraints,styleNeutral)
    if sorted(set(shenwan_instrument_industry(order_book_ids)['index_name'])) == sorted(industryConstraints.keys()) and sum([s[1] for s in industryConstraints.values()])<1:
        raise InvalidArgument("order_book_ids 权重之和小于 1, 请重新定义行业权重上下限")

    missing_industry = (set(industryConstraints)|set([] if industryNeutral is None else industryNeutral))- set(shenwan_instrument_industry(order_book_ids,date)['index_name'])
    if missing_industry:
        warnings.warn("order_book_ids 中没有股票属于{}行业, 已忽略其行业约束".format(missing_industry))

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
    portfolio_industry_label = shenwan_instrument_industry(order_book_ids, date)['index_name']
    portfolio_industry = list(portfolio_industry_label.unique())
    missing_industry = list(set(benchmark_industry_allocation.index) - set(portfolio_industry))
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
                        'book_to_price','leverage', 'liquidity', 'non_linear_size']
    if styleConstraints is None or len(styleConstraints) == 0:
        return []
    elif "*" in styleConstraints.keys():
        styleConstraints = {s:styleConstraints.get("*") for s in constrainedStyle}
    else:
        constrainedStyle = sorted(set(styleConstraints.keys())&set(constrainedStyle))
    style_factor_exposure = get_style_factor_exposure(order_book_ids,date,date,'all').xs(date,level=1)
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
    :param deviation: 偏离上下限
    :param factors: 受约束的因子列表
    :param benchmark: 基准
    :return:
    """
    constraintedStyle = ['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility', 'growth',
                        'book_to_price','leverage', 'liquidity', 'non_linear_size']

    if factors is None or deviation is None:
        return []

    benchmark_components = index_weights(benchmark,date)
    union_stocks = sorted(set(order_book_ids).union(set(benchmark_components.index)))

    style_factor_exposure = get_style_factor_exposure(union_stocks,date,date,factors='all')
    style_factor_exposure.index = style_factor_exposure.index.droplevel(1)

    constraints = []

    if "*"== factors:
        constrainted_style = constraintedStyle
    elif isinstance(factors,list):
        constrainted_style = sorted(set(constraintedStyle)&set(factors))
    else:
        raise InvalidArgument("请在风格偏离约束中指定 * 或者风格因子列表")

    benchmark_components_data = style_factor_exposure.loc[benchmark_components.index,constrainted_style]
    portfolio_data = style_factor_exposure.loc[order_book_ids,constrainted_style]

    benchmark_style_exposure = benchmark_components.dot(benchmark_components_data)

    portfolio_style_constraints = {style:(benchmark_style_exposure[style]*(1-deviation),benchmark_style_exposure[style]*(1+deviation)) for style in benchmark_style_exposure.index}

    for style in constrainted_style:
        # 当因子暴露为正时，因子暴露度*（1-偏离度<因子暴露度*（1+偏离度）;当因子暴露为负时，反因子暴露度*（1-偏离度）>因子暴露度*（1+偏离度）
        lower,upper = min(portfolio_style_constraints[style]),max(portfolio_style_constraints[style])
        constraints.append({"type":"ineq","fun":lambda x: x.dot(portfolio_data[style])-lower})
        constraints.append({"type":"ineq","fun":lambda x: upper - x.dot(portfolio_data[style])})
    return constraints

def fund_type_constraints(order_book_ids,fundTypeConstraints):
    """
    :param order_book_ids:
    :param fundTypeConstraints: {"Stock":(0,0.5),"Hybrid":(0,0.3),"Bond":(0,0.1)}
    :return:
    """
    fundTypes = [instrument.fund_type for instrument in fund.instruments(order_book_ids)]
    fundTypes = pd.Series(fundTypes,index=order_book_ids)

    constraints = []

    for fundType in fundTypeConstraints:
        fundBounds = fundTypeConstraints.get(fundType)
        fund_positions = fundTypes.index.get_indexer(fundTypes[fundTypes == fundType].index)

        constraints.append({"type":"ineq","fun":lambda x:sum(x[i] for i in fund_positions) - fundBounds[1]})
        constraints.append({'typr':"ineq","fun":lambda x : -sum(x[i] for i in fund_positions) +fundBounds[0]})
    return constraints


# def get_shenwan_data(order_book_ids,date):
#
#     shenwan_data = shenwan_instrument_industry(order_book_ids, date=date)['index_name']
#     null_data_list = list(set(order_book_ids) - set(shenwan_data.index.tolist()))
#     return shenwan_data,null_data_list


