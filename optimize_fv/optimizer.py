# -*- coding: utf-8 -*-
from sklearn.covariance import LedoitWolf
from .optimizer_toolkit import *
from collections import defaultdict
import warnings
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))
from rqdatac import *
import pandas as pd
from scipy.optimize import minimize
from enum import Enum
import datetime

ALL_STYLE_FACTORS =['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility',
       'growth', 'book_to_price', 'leverage', 'liquidity', 'non_linear_size']

def _to_constraint(reversed_index, rules):
    constraints = []
    lower_sum = upper_sum = 0
    no_constraint_exists = False

    for k, v in reversed_index.items():
        try:
            lower, upper = rules[k]
            if lower < 0 or lower > 1:
                raise InvalidArgument(u'OPTIMIZER: 约束 {} 的下限 {} 无效'.format(k, lower))
            if upper < 0 or upper > 1:
                raise InvalidArgument(u'OPTIMIZER: 约束 {} 的上限 {} 无效'.format(k, upper))
            if lower > upper:
                raise InvalidArgument(u'OPTIMIZER: 约束的下限 {} 高于上限 {}'.format(lower, upper))
            lower_sum += lower
            upper_sum += upper
            constraints.append({'type': 'ineq', 'fun': lambda x: sum(x[i] for i in v) - lower})
            constraints.append({'type': 'ineq', 'fun': lambda x: upper - sum(x[i] for i in v)})
        except KeyError:
            no_constraint_exists = True

    if lower_sum > 1:
        raise InvalidArgument(u'OPTIMIZER: 约束的下限之和大于1')
    if not no_constraint_exists and upper_sum < 1:
        raise InvalidArgument(u'OPTIMIZER: 约束的上限之和小于1')

    # constraints.append({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    return constraints

def _to_constraint2(data,rules):
    # 设置虚拟变量的约束
    constraints = []
    rules = {} if rules is None else rules
    for k,v in rules.items():

        if k in data.columns:

            lower, upper = v[0], v[1]
            if lower > upper:
                raise InvalidArgument(u'OPTIMIZER: 约束的下限 {} 高于上限 {}'.format(lower, upper))
            constraints.append({"type": "ineq", "fun": lambda x: upper - x.dot(data[k])})
            constraints.append({"type": "ineq", "fun": lambda x: -lower + x.dot(data[k])})
        else:
            warnings.warn("不存在属于{}行业中的order_book_id".format(k))
    return constraints

class FundTypeConstraint:
    def __init__(self, rules):
        self._verify_rules(rules)
        self._rules = rules

    @staticmethod
    def _verify_rules(rules):
        pass

    def to_constraint(self, order_book_ids, date):
        classifed = [rqdatac.fund.instruments(o).fund_type for o in order_book_ids]
        reversed_index = defaultdict(list)
        for i, k in enumerate(classifed):
            reversed_index[k].append(i)
        return _to_constraint(reversed_index, self._rules)

class ShenWanConstraint:
    def __init__(self, rules):
        self._verify_rules(rules)
        self._rules = rules

    @staticmethod
    def _verify_rules(rules):
        pass

    def to_constraint(self, order_book_ids, date):
        shenwan_data = pd.DataFrame(rqdatac.shenwan_instrument_industry(order_book_ids, date)['index_name'])
        shenwan_data = shenwan_data.reset_index()
        shenwan_data['values'] = 1

        dummy_variables = shenwan_data.pivot(columns="index_name",index="index",values="values").replace(np.nan,0)
        return _to_constraint2(dummy_variables,self._rules)

class StyleConstraints:
    def __init__(self,rules):
        self._verify_rules(rules)
        self._rules = rules
    @staticmethod
    def _verify_rules(rules):
        pass
    def to_constraint(self,order_book_ids,date):
        factor_data = rqdatac.get_style_factor_exposure(order_book_ids,date,date).xs(date,level=1)
        return _to_constraint2(factor_data,self._rules)

def constraintsGeneration(**kwargs):

    order_book_ids = kwargs.get("order_book_ids")
    assetsType = kwargs.get("assetsType")
    benchmark = kwargs.get("benchmark")
    date = kwargs.get("date")
    industryConstraints = kwargs.get("industryConstraints",{})
    industryNeutral = kwargs.get("industryNeutral")
    industryDeviation = kwargs.get("industryDeviation")
    styleConstraints = kwargs.get("styleConstraints",{})
    styleNeutral = kwargs.get("styleNeutral")
    styleDeviation = kwargs.get("styleDeviation")
    fundTypeConstraints = kwargs.get("fundTypeConstraints")

    constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1}]

    if assetsType == "CS":

        components_weights = index_weights(benchmark,date)

        industry_labels = shenwan_instrument_industry(components_weights.index.tolist(), date)['index_name']
        all_industry_tags = industry_labels.unique()
        def constraints_transfer(neutral_list,all_list,constraints):
            if neutral_list is None:
                pass
            elif neutral_list == "*":
                neutral_list = all_list
            elif isinstance(neutral_list,list):
                neutral_list = sorted(set(all_list)&set(neutral_list))
            else:
                raise InvalidArgument("请输入* 或者约束类别列表")
            constraints = {s: constraints.get("*") for s in all_list} if "*" in constraints.keys() else constraints
            if isinstance(neutral_list,list) and (len(list(set(neutral_list) & set(constraints))) > 0):
                raise InvalidArgument("偏离度约束和自定义约束不能重复定义")
            return constraints,neutral_list

        industryConstraints,industryNeutral = constraints_transfer(industryNeutral,all_industry_tags,industryConstraints)
        styleConstraints,styleNeutral = constraints_transfer(styleNeutral,ALL_STYLE_FACTORS,styleConstraints)

        if (industryNeutral is None) ^ (industryDeviation is None):
            raise InvalidArgument("在行业偏离度约束中必须同时指定[industryNeutral,industryDeviation]")
        elif industryNeutral is not None:
            lower_industry_deviation,upper_industry_deviation = industryDeviation[0],industryDeviation[1]
            industry_weights = pd.concat([components_weights,industry_labels],axis=1).groupby("index_name").sum().loc[industryNeutral]['weight']
            industry_deviation_rules = {s:(industry_weights[s]*(1-lower_industry_deviation),industry_weights[s]*(1+upper_industry_deviation)) for s in industry_weights.index}
        else:
            industry_deviation_rules = {}

        if (styleNeutral is None) ^ (styleDeviation is None):
            raise InvalidArgument("在风格偏离度约束中必须同时指定[styleNeutral,styleDeviation]")
        elif styleNeutral is not None:
            lower_style_deviation,upper_style_deviation = styleDeviation[0],styleDeviation[1]
            style_weights = components_weights.dot(get_style_factor_exposure(components_weights.index.tolist(), date, date).xs(date, level=1))[styleNeutral].dropna()
            style_deviation_rules = {s:(style_weights[s]*(1-lower_style_deviation),style_weights[s]*(1+upper_style_deviation)) for s in style_weights.index}
        else:
            style_deviation_rules = {}

        styleConstraints.update(style_deviation_rules)
        industryConstraints.update(industry_deviation_rules)

        industry_constraints = []
        if len(industryConstraints)>0:
            indusrty_cons = ShenWanConstraint(industryConstraints)
            industry_constraints = indusrty_cons.to_constraint(order_book_ids,date)

        style_constraints = []
        if len(styleConstraints)>0:
            style_cons = StyleConstraints(styleConstraints)
            style_constraints = style_cons.to_constraint(order_book_ids,date)

        constraints.extend(industry_constraints)
        constraints.extend(style_constraints)

    else:
        assert assetsType == "Fund"

        fund_constraints = FundTypeConstraint(fundTypeConstraints)
        fund_constraints.to_constraint(order_book_ids,date)
        constraints.extend(fund_constraints)

    return constraints

def boundsGeneration(**kwargs):
    union_stocks = kwargs.get("union_stocks")
    bounds = kwargs.get("bounds")
    order_book_ids = kwargs.get("order_book_ids")
    active_bounds = kwargs.get("active_bounds")

    if "*" in bounds.keys():
        bounds = {s: bounds.get("*") for s in order_book_ids}
    elif len(bounds) == 0:
        bounds = {s: (0, 1) for s in order_book_ids}

    not_weighted_stocks = set(union_stocks) - set(order_book_ids)
    bnds1 = {s: (0, 0) for s in not_weighted_stocks}
    bounds.update(bnds1)
    bounds.update(active_bounds)
    bounds = {s:bounds.get(s) if s in bounds else (0, 1) for s in union_stocks}

    # 进行检验，其中的not_weighted_stocks的权重不影响检验
    for k, (lower, upper) in bounds.items():
        if lower > upper:
            raise InvalidArgument(u'OPTIMIZER: 合约 {} 的下限 {} 高于上限 {}'.format(k, lower, upper))
        if lower > 1 or lower < 0:
            raise InvalidArgument(u'OPTIMIZER: 合约 {} 的下限 {} 无效'.format(k, lower))
        if upper < 0 or upper > 1:
            raise InvalidArgument(u'OPTIMIZER: 合约 {} 的上限 {} 无效'.format(k, upper))

    if sum(v[0] for v in bounds.values()) > 1:
        raise InvalidArgument(u'OPTIMIZER: bnds 下限之和大于1')
    if sum(v[1] for v in bounds.values()) < 1:
        warnings.warn("经过剔除后，bnds 上限之和小于1,已忽略指定的资产上下限")
        bnds = tuple([(0,1) for s in union_stocks])
        return bnds
        # raise InvalidArgument(u'OPTIMIZER: 经过剔除后，bnds 上限之和小于1')
    bnds = tuple([bounds.get(s) for s in union_stocks])
    return bnds

def format_active_bounds_to_general_bounds(order_book_ids,components_weights,active_bounds):

    active_bounds = {s:active_bounds.get("*") for s in order_book_ids} if "*" in active_bounds else active_bounds
    active_bounds = {k:(components_weights.get(k)*(1-l),components_weights.get(k)*(1+u)) if k in components_weights.index else (l,u) for k,(l,u) in active_bounds.items()}
    # _bounds = {s:active_bounds.get(s) if s in active_bounds else (0,1) for s in order_book_ids}

    return {k:(max(l,0),min(u,1)) for k,(l,u) in active_bounds.items()}

def assetRiskAllocation(order_book_ids,riskBudgetOptions):

    assetRank = riskBudgetOptions.get("assetRank")
    assetLabel = riskBudgetOptions.get("assetLabel")
    groupBudget = riskBudgetOptions.get("groupBudget")
    assetsType = riskBudgetOptions.get("assetsType")

    # 当指定了资产的评级，则优先使用评级进行归一化处理
    if assetRank is not None:
        try:
            assetRank = assetRank.astype(np.float64)
        except:
            print("assetRank只接受float或int类型的series")
        assetRank/=assetRank.sum()
        return assetRank
    else:
        # 只针对基金 类型进行优化权重
        if groupBudget is not None and assetLabel is None:
            if assetsType == "Fund":
                assetLabel = pd.Series([s.fund_type for s in fund.instruments(order_book_ids)], index=order_book_ids)
                groupBudget = {key: groupBudget.get(key, 0) for key in set(assetLabel)}
                assetLabel.replace(groupBudget, inplace=True)
                return assetLabel.groupby(assetLabel).transform(lambda x: x.name / len(x))
            else:
                raise Exception("对股票进行风险预算时需要指定 assetRank 或者 assetLabel")
        #     指定了标签和风险预算时
        elif groupBudget is not None and assetLabel is not None:
            groupBudget = {key: groupBudget.get(key, 0) for key in set(assetLabel)}
            assetLabel.replace(groupBudget, inplace=True)
            return assetLabel.groupby(assetLabel).transform(lambda x: float(x.name) / len(x))
        # 给定数据不够，进行风险平价
        else:
            return pd.Series(1. / len(order_book_ids), index=order_book_ids)

class OptimizeMethod(Enum):
    INDICATOR_MAXIMIZATION = maximizing_series
    RETURN_MAXIMIZATION = maximizing_series
    VOLATILITY_MINIMIZATION = volatility
    TRACKING_ERROR_MINIMIZATION = trackingError
    MEAN_VARIANCE = mean_variance
    RISK_PARITY = risk_parity
    RISK_BUDGETING = risk_budgeting

def objectiveFunction(x, function, **kwargs):
    return function(x, **kwargs)

def covarianceEstimation(daily_returns,cov_estimator):
    lw = LedoitWolf()
    if cov_estimator == "shrinkage":
        return lw.fit(daily_returns).covariance_
    elif cov_estimator == "empirical":
        return daily_returns.cov()
    elif cov_estimator == "multifactor":
        # TODO 因子协方差矩阵
        return daily_returns.cov()
    else:
        raise Exception("协方差矩阵类型为[shrinkage,empirical,multifactor]")

def risk_parity_gradient(x, **kwargs):
    c_m = kwargs.get("c_m")
    c = 15
    return np.multiply(2, np.dot(c_m, x)) - np.multiply(c, np.reciprocal(x))

def min_variance_gradient(x, **kwargs):
    c_m = kwargs.get("c_m")
    return np.multiply(2, np.dot(c_m, x))

def mean_variance_gradient(x, **kwargs):
    c_m = kwargs.get("c_m")
    annualized_return = kwargs.get("series")
    risk_aversion_coefficient = kwargs.get("risk_aversion_coefficient")
    return np.asfarray(np.multiply(2*risk_aversion_coefficient, np.dot(x, c_m)).transpose()
                       - annualized_return).flatten()

def maximum_series_gradient(x,**kwargs):
    series = kwargs.get("series")
    return -np.asfarray(series)


def getGradient(x,method,**kwargs):
    if method is OptimizeMethod.RISK_PARITY:
        return risk_parity_gradient(x,**kwargs)
    elif method is OptimizeMethod.MEAN_VARIANCE:
        return mean_variance_gradient(x,**kwargs)
    elif method is OptimizeMethod.VOLATILITY_MINIMIZATION:
        return min_variance_gradient(x,**kwargs)
    elif method is OptimizeMethod.INDICATOR_MAXIMIZATION or method is OptimizeMethod.RETURN_MAXIMIZATION:
        return maximum_series_gradient(x,**kwargs)
    else:
        pass


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
        stamp_duty_cost = pd.Series([0.001 if stampDuty and s in short_position.index else 0 for s in all_assets],index=all_assets)

        commission = pd.Series([commissionRatio]*len(all_assets),index=all_assets)

        merged_data = pd.DataFrame({"交易股数":amounts,"交易金额":positions,"印花税费率":stamp_duty_cost,"佣金费率":commission,"市场冲击成本":impact_cost,"出清时间（天）":clearing_time,"交易方向":sides,"自定义费率":customizedCost.loc[all_assets].replace(np.nan,0)})
        merged_data['总交易费率'] = commission+impact_cost+merged_data['自定义费率']+stamp_duty_cost
    if output:
        return merged_data
    else:
        if assetType == "CS":
            return merged_data['总交易费率'].dot(merged_data['交易金额'])*(252/holdingPeriod)/merged_data['交易金额'].sum()
        else:
            return merged_data['交易金额'].dot(merged_data['总费率'])*(252/holdingPeriod)/merged_data['交易金额'].sum()

MIN_OPTIMIZE_DATE = datetime.datetime(2005, 7, 1)
MIN_OPTIMIZE_DATE_STR = '2005-07-01'

def portfolio_optimize(order_book_ids, date, method=OptimizeMethod.VOLATILITY_MINIMIZATION, cov_estimator="empirical",
                        window=126, bounds=None,industryConstraints=None, styleConstraints=None,fundTypeConstraints = None,
                            annualized_return = None,indicator_series=None,returnRiskRatio = None,
                          active_bounds = None,  max_iteration = 1000,tol = 1e-8,quiet = False,filter=None,
                           benchmarkOptions=None,riskBudegtOptions=None,transactionCostOptions=None):
    """
    :param order_book_ids:组合中的合约代码，支持股票及基金（注意组合中不能同时包含股票和基金）
    :param date: 优化日期 如"2018-06-20"
    :param method 进行优化的目标函数 详见 OptimizeMethod
    :param cov_estimator: str 目标函数优化时的协方差矩阵
            可选参数：
                -"shrinkage" 默认值; 对收益的协方差矩阵进行收缩矩阵
                -"empirical" 计算个股收益的经验协方差矩阵
                -"multifactor" 因子协方差矩阵
    :param window: int 预期协方差矩阵轨迹的历史价格序列长度 默认取前126个交易日
    :param bounds: dict 个股头寸的上下限
                        1、{'*': (0, 0.03)},则所有个股仓位都在0-3%之间;
                        2、{'601099.XSHG':(0,0.3),'601216.XSHG':(0,0.4)},则601099的仓位在0-30%，601216的仓位在0-40%之间
    :param industryConstraints:dict 行业的权重上下限
                        1、{'*': (0, 0.03)},则所有行业权重都在0-3%之间;
                        2、{'银行':(0,0.3),'房地产':(0,0.4)},则银行行业的权重在0-30%，房地产行业的权重在0-40%之间
    :param styleConstraints:dict 风格约束的上下界
                        1、{"*":(1,3)},则所有的风格暴露度在1-3之间
    :param fundTypeConstraints: 基金约束
    :param annualized_return: 预期年化收益,用于均值方差模型 和 预期收益最大化
    :param indicator_series: 最大化指标优化中指定的指标序列. 指标应为浮点型变量(例如个股EP/个股预期收益/个股夏普率)

    :param returnRiskRatio：均值方差中收益和风险的偏好比例 tuple(0.7,0.3)
    :param active_bounds: 在目标函数为最小化跟踪误差时的 主动权重上下限，格式与bounds形式一样

    :param max_iteration:优化器最大迭代次数
    :param tol: 优化终止的变化步长
    :param quiet: 是否在回测日志中打印优化器对组合进行预筛选的提示信息，默认打印(False)

    :param benchmarkOptions dict,优化器的可选参数，该字典接受如下参数选项
                - benchmark: string 组合跟踪的基准指数, 默认为"000300.XSHG"
                - cov_estimator:约束时所使用的协方差矩阵
                        可选参数
                        -"shrinkage" 默认值; 对收益的协方差矩阵进行收缩矩阵
                        -"empirical" 对收益进行经验协方差矩阵计算
                - trackingErrorThreshold: float 可容忍的最大跟踪误差
                - industryNeutral: list 设定投资组合相对基准行业存在偏离的行业
                - industryDeviation tuple 投资组合的行业权重相对基准行业权重容许偏离度的上下界
                - styleNeutral: list 设定投资组合相对基准行业存在偏离的barra风格因子
                - styleDeviation tuple 投资组合的风格因子权重相对基准风格容许偏离度的上下界
                - industryMatching:boolean 是否根据基准成分股进行配齐
    :param riskBudgetOptions: dict,优化器可选参数,该字典接受如下参数
                - riskMetrics: 风险预算指标 默认为波动率 str volatility,tracking_error
                - assetRank: 资产评级
                - groupBudget: 风险预算
                - assetLabel: 资产分类

    :param transactionCostOptions: dict 优化器可选参数，该字典接受如下参数
                - initialCapital: float 即账户中用于构建优化组合的初始资金（元）。用于计算构建优化组合所需要承担的交易费用
                - currentPositions:账户当前仓位信息（股票以股数为单位，现金以元为单位，公募基金以份额为单位）。对于股票交易费用，会以账户当前股票仓位、现金余额、及当天收盘价计算账户总权益，再计算构建优化组合所需承担的交易费用；对于基金申赎费用，会以账户当前公募基金仓位、现金余额、及当天公募基金公布的单位净值计算账户总权益，再计算构建优化FOF组合所需承担的交易费用。
                - holdingPeriod:持仓周期，默认为21个交易日（约一个月）
                - commission:是否考虑股票交易佣金。默认佣金费率为万分之八（0.008）。交易佣金费率可通过参数commissionRatio进行调整
                - commissionRatio:指定股票交易佣金费率。默认为0.0008
                - subscriptionRedemption:是否考虑基金申赎费用（不考虑认购费）。各类基金的默认申赎费率见下表。对于申购费用，默认采用前端收费(4)，费用按外扣法计算(5)。申赎费率可通过参数subRedRatio进行调整
                - subRedRatio:指定基金申赎费率。各类基金的默认申赎费率见下表。假定用户希望优化中对设定股票型基金的申购费率为1%，赎回费率调整为0.5%，则可传入字典型参数如下：{‘Stock’: (0.01, 0.005)}
                - stampDuty:是否考虑股票交易印花税，税率固定为0.1%（只在卖出股票时征收）。基金申赎不涉及印花税。
                - marketImpact:股票交易中是否考虑市场冲击成本（计算说明见附录）。基金申赎不考虑该成本
                - marketImpactRatio:指定市场冲击成本系数。默认为1
                - customizedCost:用户自定义交易费用。可与上述其它费用加和得到总费用。
                - cashPosition: 指定现金的仓位
                - output:是否同时返回个股优化权重和交易费用。默认为 False。若取值为True，则返回相关交易费用相关信息（佣金、印花税、市场冲击成本、申赎费用、日波动率、出清时间）。
    :return: pandas.Series优化后的个股权重, 传入order_book_id 中过滤的股票:{"次新股":subnew_stocks, "停牌股":suspended_stocks,"ST类股票":st_stocks,"退市股票":delisted_stocks}
    """
    benchmarkOptions = {} if benchmarkOptions is None or (method is OptimizeMethod.RISK_BUDGETING) else benchmarkOptions
    benchmark = benchmarkOptions.get("benchmark", "000300.XSHG")
    cov_estimator_in_option = benchmarkOptions.get('cov_estimator')
    trackingErrorThreshold = benchmarkOptions.get("trackingErrorThreshold")
    industryNeutral, industryDeviation = benchmarkOptions.get('industryNeutral'), benchmarkOptions.get('industryDeviation')
    styleNeutral, styleDeviation = benchmarkOptions.get("styleNeutral"), benchmarkOptions.get("styleDeviation")
    industryMatching = benchmarkOptions.get("industryMatching")

    riskBudgetOptions = {} if riskBudegtOptions is None else riskBudegtOptions
    bounds = {} if bounds is None else bounds
    filter = {} if filter is None or not isinstance(filter,dict) else filter
    active_bounds = {} if active_bounds is None else active_bounds
    industryConstraints = {} if industryConstraints is None else industryConstraints
    styleConstraints = {} if styleConstraints is None else styleConstraints

    default_filter = {"suspend_filter":True,"subnew_filter":window,"st_filter":False}
    default_filter.update(filter)
    filter = default_filter

    if isinstance(date,str) and date <MIN_OPTIMIZE_DATE_STR or (isinstance(date,datetime.datetime)) and date <MIN_OPTIMIZE_DATE:
        raise InvalidArgument("日期不能早于2005-07-01（之前没有足够的数据）")

    # 只有当优化过程中涉及跟踪误差, benchmark才是有效参数，这里设置新的_benchmark是将构建股票列表 和 约束 两者所用的benchmark分开
    _benchmark = benchmark if (method is OptimizeMethod.TRACKING_ERROR_MINIMIZATION or isinstance(trackingErrorThreshold,float) or (riskBudgetOptions.get("riskMetrics") == "tracking_error")) else None
    assetsType = ensure_same_type_instruments(order_book_ids)

    if _benchmark is not None and assetsType == "Fund":
        raise InvalidArgument("OPTIMIZER: [基金] 不支持 包含跟踪误差 约束 的权重优化 ")
    if (not isinstance(annualized_return,pd.Series)) and (method is OptimizeMethod.MEAN_VARIANCE or method is OptimizeMethod.RETURN_MAXIMIZATION):
        raise InvalidArgument("OPTIMIZER: 均值方差和预期收益最大化时需要指定 annualized_return [预期年化收益]")
    if (not isinstance(indicator_series,pd.Series)) and (method is OptimizeMethod.INDICATOR_MAXIMIZATION):
        raise InvalidArgument("OPTIMIZER: 指标序列最大化时需要指定 indicator_series [指标序列]")
    if method is OptimizeMethod.MEAN_VARIANCE and not (isinstance(returnRiskRatio,tuple) and (np.round(returnRiskRatio[0]+returnRiskRatio[1]) == 1)):
        raise InvalidArgument("OPTIMIZER: 在均值方差模型中请指定returnRiskRatio且收益和风险的比例之和=1")
    if not ((method is OptimizeMethod.RETURN_MAXIMIZATION or method is OptimizeMethod.INDICATOR_MAXIMIZATION) and benchmarkOptions.get("trackingErrorThreshold") is None):
        if not filter.get("suspend_filter"):
            raise InvalidArgument("此优化器涉及协方差矩阵计算，为避免空值对优化器的影响，请过滤停牌股票，将[suspend_filter]置为True")
        if not (isinstance(filter.get("subnew_filter"),int) and filter.get("subnew_filter")>=window):
            raise InvalidArgument("此优化器涉及协方差矩阵计算，为避免空值对优化器的影响，请过滤次新股，将[subnew_filter]设为>=window的整数")

    order_book_ids, union_stocks, subnew_stocks,delisted_stocks,suspended_stocks,st_stocks = assetsListHandler(filter,date=date,order_book_ids=order_book_ids,window=window,benchmark=_benchmark,assetsType=assetsType)


    if not quiet:
        if len(suspended_stocks) > 0:
            warnings.warn("OPTIMIZER:已剔除停牌股票：{}".format(suspended_stocks))
        if len(subnew_stocks)>0:
            warnings.warn("OPTIMIZER:已剔除次新股：{}".format(subnew_stocks))

    if len(order_book_ids)<=2:
        raise OptimizationFailed("OPTIMIZER: 经过剔除后，合约数目不足2个")

    components_weights = index_weights(benchmark, date).reindex(union_stocks).replace(np.nan,0) if benchmark is not None else None
    active_bounds = format_active_bounds_to_general_bounds(order_book_ids, components_weights, active_bounds)
    constraints = constraintsGeneration(order_book_ids = union_stocks,assetsType = assetsType,benchmark=benchmark,
                          date=date,industryConstraints=industryConstraints,industryNeutral=industryNeutral,
                          industryDeviation=industryDeviation,styleConstraints=styleConstraints,styleNeutral=styleNeutral,
                          styleDeviation=styleDeviation,fundTypeConstraints=fundTypeConstraints)
    bnds = boundsGeneration(union_stocks =union_stocks,bounds =bounds,order_book_ids =order_book_ids,active_bounds=active_bounds)

    optimize_with_cons_or_bnds = (len(constraints)>1 or len(bounds)>0)
    risk_parity_without_cons_and_bnds = (not optimize_with_cons_or_bnds) and method is OptimizeMethod.RISK_PARITY

    if method is OptimizeMethod.RISK_BUDGETING or risk_parity_without_cons_and_bnds:
        bnds = tuple([(1e-6,None) if _bnds[0] == 0 else _bnds for _bnds in bnds])

    start_date = trading_date_offset(date,-window-1)
    if assetsType == 'CS':
        daily_returns = get_price(union_stocks, start_date, date, fields="close").pct_change().dropna(how='all').iloc[-window:]
    else:
        assert assetsType == "Fund"
        daily_returns = fund.get_nav(order_book_ids,start_date,date,fields="acc_net_value").pct_change().dropna(how="all").iloc[-window:]

    #   目标函数的协方差矩阵 在 收益最大化/指标最大化中不涉及，并且 仅在目标函数包含跟踪误差时该矩阵才包含 非分配权重 的基准成分股
    c_m = None
    if not (method is OptimizeMethod.RETURN_MAXIMIZATION and method is OptimizeMethod.INDICATOR_MAXIMIZATION):
        c_m = covarianceEstimation(daily_returns=daily_returns,cov_estimator=cov_estimator) if method is OptimizeMethod.TRACKING_ERROR_MINIMIZATION or riskBudgetOptions.get("riskMetrics") == "tracking_error" else covarianceEstimation(daily_returns=daily_returns[order_book_ids],cov_estimator=cov_estimator)

    x0 = (pd.Series(1, index=order_book_ids) / len(order_book_ids)).reindex(union_stocks).replace(np.nan, 0).values

    if isinstance(trackingErrorThreshold,float):
        cons_c_m = covarianceEstimation(daily_returns=daily_returns,cov_estimator=cov_estimator_in_option)

        trackingErrorOptions = {"c_m": cons_c_m, "union_stocks": union_stocks,"index_weights":components_weights.values}
        constraints.append({"type": "ineq", "fun": lambda x: -trackingError(x,**trackingErrorOptions) + trackingErrorThreshold})

    riskMetrics = riskBudgetOptions.get("riskMetrics", "volatility")

    annualized_return = annualized_return.loc[order_book_ids].replace(np.nan,0) if isinstance(annualized_return,pd.Series) else None

    # 以下是优化目标函数中可能涉及到的参数
    input_series = annualized_return if (method is OptimizeMethod.RETURN_MAXIMIZATION or method is OptimizeMethod.MEAN_VARIANCE) else indicator_series
    kwargs = {"c_m": c_m,  "series": input_series,"index_weights":components_weights.values,
              "riskMetrics":riskMetrics,"with_cons":optimize_with_cons_or_bnds}

    # 在风险预算中需要先计算每个股票的风险预算
    if method is OptimizeMethod.RISK_BUDGETING:
        riskBudgetOptions.update({"assetsType":assetsType})
        riskBudgets = assetRiskAllocation(order_book_ids, riskBudgetOptions)
        riskBudgets = riskBudgets.reindex(union_stocks).replace(np.nan,0)
        constraints = []
        if riskMetrics == "volatility":
            constraints.append({"type": "ineq", "fun": lambda x: riskBudgets.dot(np.log(x)) - 13})
        elif riskMetrics == "tracking_error":
            if assetsType == "CS":
                kwargs["riskBudgets"] = riskBudgets
            else:
                raise InvalidArgument("使用跟踪误差进行风险预算时仅限股票使用")
        else:
            raise InvalidArgument("风险预算中的风险度量仅 支持 [volatility,tracking_error],请在riskBudgetOptions中riskMetrics重新指定")

    # 在均值方差模型中，先计算风险厌恶系数
    if method is OptimizeMethod.MEAN_VARIANCE:
        returnRiskRatio = returnRiskRatio[0]/returnRiskRatio[1]
        risk_aversion_coefficient = (x0.dot(annualized_return))/(volatility(x0,**kwargs)*returnRiskRatio)
        kwargs.update({"risk_aversion_coefficient":risk_aversion_coefficient})

    # 当transactionCostOptions非空时，即用户希望在优化中考虑成本分析，需提供initialCapital或者currentPositions
    include_transaction_cost = False
    if transactionCostOptions is not None :
        initialCapital = transactionCostOptions.get("initialCapital")
        currentPositions = transactionCostOptions.get("currentPositions")
        if (initialCapital is None and (not isinstance(currentPositions,pd.Series))):
            raise Exception("考虑交易费用时, [initialCapital] 初始金额 [currentPositions] 当前持仓 必须指定其中之一 ")
        else :
            include_transaction_cost = True

    def _objectiveFunction(x):
        if include_transaction_cost:
            return objectiveFunction(x, method, **kwargs) - calcTransactionCost(order_book_ids,x,date,assetsType, transactionCostOptions)
        else:
            return objectiveFunction(x,method,**kwargs)

    options = {'maxiter': max_iteration,"ftol":tol}
    kwargs_gradient = {"c_m":c_m,"series":input_series}

    def _get_gradient(x):
        _fun = getGradient(x, method, **kwargs_gradient)
        return _fun

    # 默认使用最小二乘，在不带有约束的风险平价除外
    iter_method = "SLSQP"
    if risk_parity_without_cons_and_bnds:
        constraints = None
        iter_method = "L-BFGS-B"
        options = {'maxiter': max_iteration}
    # if (method is OptimizeMethod.RISK_BUDGETING and riskMetrics == "tracking_error") or method is OptimizeMethod.MEAN_VARIANCE:
    #     options = {'maxiter': max_iteration}

    while True:
        if (method in [OptimizeMethod.VOLATILITY_MINIMIZATION,OptimizeMethod.MEAN_VARIANCE,OptimizeMethod.RETURN_MAXIMIZATION,OptimizeMethod.INDICATOR_MAXIMIZATION] or risk_parity_without_cons_and_bnds) and include_transaction_cost:
            optimization_results = minimize(_objectiveFunction, x0, bounds=bnds, jac=_get_gradient,constraints=constraints, method=iter_method, options=options)
        else:
            optimization_results = minimize(_objectiveFunction, x0, bounds=bnds, jac=None,constraints=constraints, method=iter_method, options=options)

        if optimization_results.success:
            optimized_weight = pd.Series(optimization_results['x'], index=union_stocks).reindex(order_book_ids)
            optimized_weight /= optimized_weight.sum()
            if industryMatching:
                # 获得未分配行业的权重与成分股权重
                unallocatedWeight, supplementStocks = benchmark_industry_matching(order_book_ids, benchmark, date)
                optimized_weight = pd.concat([optimized_weight * (1 - unallocatedWeight), supplementStocks])
            return optimized_weight,{"次新股":subnew_stocks, "停牌股":suspended_stocks,"ST类股票":st_stocks,"退市股票":delisted_stocks}
        elif optimization_results.status == 8:
            if options['ftol'] >= 1e-3:
                raise Exception(u'OPTIMIZER: 优化无法收敛到足够精度')
            options['ftol'] *=10
        else:
            raise Exception(optimization_results.message)


# COMMENTS:

# 行业约束与优化器1.0有些不同
# 更改使用点乘形式约束行业权重
# 亦可将下段comment恢复求和形式的约束

# class ShenWanConstraint:
#     def __init__(self, rules):
#         self._verify_rules(rules)
#         self._rules = rules
#
#     @staticmethod
#     def _verify_rules(rules):
#         pass
#
#     def to_constraint(self, order_book_ids, date):
#         series = rqdatac.shenwan_instrument_industry(order_book_ids, date)['index_name']
#         classified = [series.get(o, '__NONE__') for o in order_book_ids]
#         reversed_index = defaultdict(list)
#         for i, k in enumerate(classified):
#             reversed_index[k].append(i)
#         return _to_constraint(reversed_index, self._rules)

# [constraintsGeneration]
# 将传入的约束条件格式化输出统一格式并且排除错误
# 这里指的统一格式是指例如规定 industryNeutral=['银行']，industryDeviation=(0.1,0.2), 银行在benchmark中的权重为0.3，格式化后银行权重为(0.3*(1-0.1),0.3*(1+0.2))
# 将偏离度约束的[xxxNeutral,xxxDeviation]转换成标准格式,即上下限均为分配权重e.g {"银行":(0,0.2)}/{"beta":(0,0.3)}
# 检验的内容包括：
# 1/在行业(风格)的自定义和偏离存在重复约束
# 2/在偏离约束中的[industryNeutral(styleNeutral)]并非有效列表
# 3/在偏离度约束中xxxNeutral和xxxDeviation只指定了一个

# [boundsGeneration]
# 以下生成规范格式的bounds，并且检验bounds的合理性
# union_stocks仅在优化条件中涉及跟踪误差时与order_book_ids不相符
# order_book_ids 为最终需要分配权重的股票，对于不分配权重的股票由于其在计算跟踪误差波动率时也包括在其中，因此定义其权重为(0,0)
# 检验的内容包括：
# 1/个股定义的上限<下限
# 2/下限<0 或者 下限>1
# 3/上限>1
# 4/下限之和<1
# 5/上限之和>1

# [assetRiskAllocation]
# 根据传入的风险预算字典进行单个资产风险预算分配
# 按如下逻辑进行风险分配：
# 1/若用户输入单个资产评级序列，则按评级进行风险分配；
# 2/若用户输入各类资产的风险分配字典，则按给定值对各类资产分配风险，再在每类资产内部对每个资产分配相同风险；
# 3/若用户同时提供单个资产评级序列和各类资产风险分配字典，则优先按单个资产评级进行风险分配
# 4/若用户未提供单个资产评级序列或各类资产风险分配字典，则按对每个资产分配相同风险进行计算（即风险平价）

# [format_active_bounds_to_general_bounds]
# 将主动权重转换成标准权重上下限
# 对于基准权重不为0的股票，其权重就是主动权重
# 对于未定义主动权重的，默认上下限(0,1)