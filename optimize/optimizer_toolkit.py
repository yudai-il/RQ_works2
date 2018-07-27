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

def get_subnew_assets(order_book_ids,date,N,type="CS"):
    """
    # 获得某日上市小于N天的次新股
    :param stocks: list 股票列表
    :param date: str eg. "2018-01-01"
    :param N: int 次新股过滤的阈值
    :return: list 列表中的次新股
    """
    if type == "CS":
        return [s for s in order_book_ids if len(get_trading_dates(instruments(s).listed_date,date))<=N]
    elif type == "Fund":
        return [s for s in order_book_ids if len(get_trading_dates(fund.instruments(s).listed_date, date)) <= N]

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
    start_date = pd.Timestamp(end_date) - np.timedelta64(N * 2, "D")
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

def assetsListHandler(**kwargs):
    benchmark = kwargs.get("benchmark")
    date = kwargs.get("date")
    order_book_ids = kwargs.get("order_book_ids")
    window = kwargs.get("window")
    assetsType = kwargs.get("assetsType")

    if assetsType == "CS":

        benchmark_components = index_components(benchmark, date=date) if benchmark is not None else []
        union_stocks = sorted(set(benchmark_components).union(set(order_book_ids)))

        subnew_stocks = get_subnew_assets(union_stocks, date, window)
        suspended_stocks = get_suspended_stocks(union_stocks, date, window)
        union_stocks = sorted(set(union_stocks) - set(subnew_stocks) - set(suspended_stocks))

        subnew_stocks = set(subnew_stocks)&set(order_book_ids)
        suspended_stocks = set(suspended_stocks)&set(order_book_ids)

        order_book_ids = sorted(set(union_stocks)&set(order_book_ids))

        return order_book_ids,union_stocks,suspended_stocks,subnew_stocks
    else:
        assert assetsType == "Fund"
        subnew_funds = get_subnew_assets(order_book_ids, date, window,type="Fund")
        order_book_ids = sorted(set(order_book_ids)-set(subnew_funds))
        return order_book_ids,order_book_ids,[],subnew_funds

def assetsDistinguish(order_book_ids):
    stocksInstruments = instruments(order_book_ids)
    fundsInstruments = fund.instruments(order_book_ids)

    stocksInstrumentsTypes = ([instrument.type for instrument in stocksInstruments])
    cons_all_funds = len(fundsInstruments) == len(order_book_ids)
    cons_all_stocks = stocksInstrumentsTypes.count("CS") == len(order_book_ids)

    if not (cons_all_funds or cons_all_stocks):
        raise Exception("输入order_book_ids必须全为股票或者基金")
    assetsType = "Fund" if cons_all_funds else "CS"
    return assetsType


def trackingError(x,**kwargs):
    """
    跟踪误差约束
    :param x: 权重
    :param union_stocks:股票并集
    :param index_weights: 基准权重
    :param covMat: 协方差矩阵
    :return: float
    """
    union_stocks, covMat,_index_weights = kwargs.get("union_stocks"),kwargs.get("covMat"),kwargs.get("index_weights")
    # vector of deviations
    X = x - _index_weights
    result = np.sqrt(np.dot(np.dot(X, covMat * 252),X))
    return result

def volatility(x,**kwargs):
    covMat = kwargs.get("covMat")
    """
    计算投资组合波动率
    :param x:
    :param covMat:
    :return:
    """
    return np.dot(np.dot(x,covMat*252),x)

def mean_variance(x,**kwargs):

    annualized_return = kwargs.get("annualized_return")
    risk_aversion_coefficient = kwargs.get("risk_aversion_coefficient")

    if not isinstance(annualized_return,pd.Series):
        raise Exception("在均值方差优化中请指定 预期收益")
    portfolio_volatility = volatility(x,**kwargs)
    return -x.dot(annualized_return) + np.multiply(risk_aversion_coefficient/2,portfolio_volatility)

def maximizing_return(x,**kwargs):
    annualized_return = kwargs.get("annualized_return")
    return -x.dot(annualized_return)


def maximizing_indicator(x,**kwargs):
    indicator_series = kwargs.get("indicator_series")
    return -x.dot(indicator_series)

def risk_budgeting(x,**kwargs):
    riskMetrics = kwargs.get("riskMetrics")
    if riskMetrics == "volatility":
        return np.sqrt(volatility(x, **kwargs))
    else:
        assert riskMetrics == "tracking_error"
        return trackingError(x,**kwargs)

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
        res =  np.dot(x, np.dot(c_m, x)) - c * sum(np.log(x))
        # print(res)
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
    industries_labels = missing_industryLabel_handler(order_book_ids,date)

    if industryConstraints is None or len(industryConstraints) == 0:
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
    # 检验行业/风格的 自定义约束 和 偏离约束不能存在重复定义
    def _constraintsCheck(constraints, neutral):
        if constraints is None or neutral is None:
            pass
        elif ("*" in constraints.keys() and len(neutral) > 0) or ("*" == neutral and len(constraints) > 0):
            raise Exception("自定义约束 和 偏离约束 不能存在重叠部分")
        elif set(constraints) & set(neutral):
            raise Exception("自定义约束 和 偏离约束 不能存在重叠部分")

    def _boundsCheck(bounds):
        bounds = {} if bounds is None else bounds
        bounds = bounds
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
            raise Exception("OPTIMIZER: 约束的下限之和大于1")
    #     对于行业
    _boundsCheck(bounds)
    _boundsCheck(industryConstraints)

    upperCumsum = np.sum(s[1] for s in bounds.values())
    #
    if upperCumsum<1:
        raise InvalidArgument(u'OPTIMIZER: 约束的上限之和小于1')

    _constraintsCheck(industryConstraints,industryNeutral)
    _constraintsCheck(styleConstraints,styleNeutral)
    if sorted(set(shenwan_instrument_industry(order_book_ids)['index_name'])) == sorted(industryConstraints.keys()) and sum([s[1] for s in industryConstraints.values()])<1:
        raise Exception("order_book_ids 权重之和小于 1, 请重新定义行业权重上下限")

    missing_industry = set(industryConstraints)- set(missing_industryLabel_handler(order_book_ids,date))
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
    if styleConstraints is None or len(styleConstraints) == 0:
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

    constraintedStyle = ['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility', 'growth',
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
        constrainted_style = constraintedStyle
    elif isinstance(factors,list):
        constrainted_style = sorted(set(constraintedStyle)&set(factors))
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

# 2018-07-09 calculating var and cvar


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



def calcTransactionCost(order_book_ids,x,date,assetType,transactionOptions):

    transactionOptions = {s[0]: s[1] for s in transactionOptions.items() if s[1] is not None}

    initialCapital = transactionOptions.get("initialCapital")
    currentPositions = transactionOptions.get("currentPositions")
    holdingPeriod = transactionOptions.get("holdingPeriod",21)
    commission = transactionOptions.get("commission",True)
    subscriptionRedemption = transactionOptions.get("subscriptionRedemption")
    stampDuty = transactionOptions.get("stampDuty")
    marketImpact = transactionOptions.get("marketImpact",True)
    commissionRatio = transactionOptions.get("commissionRatio",0.0008)
    subRedRatio = transactionOptions.get("subRedRatio",{})
    marketImpactRatio = transactionOptions.get("marketImpactRatio",1)
    customizedCost = transactionOptions.get("customizedCost")
    cashPosition = transactionOptions.get("cashPosition",0)
    output = transactionOptions.get("output")

    commissionRatio = commissionRatio if commission else 0
    marketImpactRatio = marketImpactRatio if marketImpact else 0

    defaultSubRedRatio = {"Stock": (0.015, 0.005), "Bond": (0.006, 0.005),
                   "Hybrid": (0.015, 0.005), "StockIndex": (0.015, 0.005),
                   "BondIndex": (0.006, 0.005), "Related": (0.006, 0.005),
                   "QDII": (0.016, 0.005), "ShortBond": (0, 0),
                   "Money": (0, 0), "Other": (0, 0)}

    subRedRatio = {} if subRedRatio is None else subRedRatio
    defaultSubRedRatio.update(subRedRatio)
    subRedRatio = defaultSubRedRatio if subscriptionRedemption else {}

    # 获得当前持仓的最新价格
    cash = currentPositions['cash'] if currentPositions is not None and "cash" in currentPositions.index else 0
    holding_assets = currentPositions[~(currentPositions.index == "cash")].index.tolist() if currentPositions is not None else []
    all_assets = sorted(set(holding_assets) | set(order_book_ids))
    latest_price = fund.get_nav(all_assets,start_date=date,end_date=date,fields="adjusted_net_value").iloc[0] if assetType == "Fund" else get_price(all_assets,start_date=date,end_date=date,fields="close",adjust_type="pre").iloc[0]

    # 获得当前持有的权益
    market_value = (latest_price*currentPositions[~("cash" == currentPositions.index)]).replace(np.nan,0) if currentPositions is not None else latest_price*0
    #  总权益
    total_equity = initialCapital if currentPositions is None else market_value.sum()+cash
    # 用户非现金权益

    allocated_equity = total_equity*(1-cashPosition)
    # 获得权益的变化
    x = pd.Series(x,index=order_book_ids).reindex(all_assets).replace(np.nan,0)
    x/=x.sum()

    position_delta = (x*allocated_equity) - market_value
    long_position = position_delta[position_delta>0]
    short_position = -position_delta[position_delta<0]

    customizedCost = pd.Series(0,index=all_assets) if not isinstance(customizedCost,pd.Series) else customizedCost

    # 对于基金
    if assetType == "Fund":
        item_names = ['申赎费用', '申赎费率', '申赎份额']
        subscription_ratios = pd.Series({s.order_book_id:subRedRatio.get(s.fund_type,(0,0))[0]for s in fund.instruments(long_position.index.tolist())})
        redemption_ratios = pd.Series({s.order_book_id:subRedRatio.get(s.fund_type,(0,0))[1] for s in fund.instruments(short_position.index.tolist())})

        # 申购时采用前端收费，外扣法计算方式
        net_subscription_value = long_position/(1+subscription_ratios)
        subscription_fees = long_position - net_subscription_value
        subscription_costs = pd.concat([subscription_fees,subscription_fees/long_position,net_subscription_value/latest_price.loc[long_position.index]],axis=1)
        subscription_costs.columns = item_names
        subscription_costs['交易方向'] = "申购"

        #  赎回费用
        redemption_costs = pd.concat([redemption_ratios*short_position,redemption_ratios,short_position/latest_price.loc[short_position.index]],axis=1)
        redemption_costs.columns = item_names
        redemption_costs['交易方向'] = '赎回'

        merged_data = pd.concat([subscription_costs,redemption_costs])
        merged_data['交易金额'] = pd.concat([long_position, short_position])
        merged_data['自定义费率'] = customizedCost.loc[all_assets].replace(np.nan,0)
        merged_data['总费率'] = merged_data['申赎费率']+merged_data['自定义费率']
        merged_data['自定义费用'] = merged_data['自定义费率']*merged_data['交易金额']

    else:
        assert assetType == "CS"
        # 对于股票
        data = get_price(all_assets,pd.Timestamp(date)-np.timedelta64(400,"D"),date,fields=["close",'volume'],adjust_type="pre").iloc[-253:]
        close_price = data['close']
        volume = data['volume']
        daily_volatility = close_price.pct_change().dropna(how="all").std()
        avg_volume = volume.iloc[-5:].mean()

        positions = pd.concat([short_position,long_position])
        positions.name = "交易金额"

        sides = pd.Series(["卖" if s in short_position.index else "买" for s in all_assets],index=all_assets)
        amounts = positions/latest_price

        clearing_time = amounts / avg_volume
        impact_cost = marketImpactRatio * daily_volatility * clearing_time**(1/2)
        stamp_duty_cost = pd.Series([0.001 if stampDuty and s in short_position.index  else 0 for s in all_assets],index=all_assets)

        commission = pd.Series([commissionRatio]*len(all_assets),index=all_assets)

        merged_data = pd.DataFrame({"交易股数":amounts,"交易金额":positions,"印花税费率":stamp_duty_cost,"佣金费率":commission,"市场冲击成本":impact_cost,"出清时间（天）":clearing_time,"交易方向":sides,"自定义费率":customizedCost.loc[all_assets].replace(np.nan,0)})
        merged_data['总交易费用'] = commission+impact_cost+merged_data['自定义费率']+stamp_duty_cost
    if output:
        return merged_data
    else:
        if assetType == "CS":
            return merged_data['总交易费用']*(252/holdingPeriod)
        else:
            return merged_data['总费率']*(252/holdingPeriod)




