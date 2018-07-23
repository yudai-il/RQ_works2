from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from .optimizer_toolkit import *
from enum import Enum


class OptimizeMethod(Enum):
    INDICATOR_MAXIMIZATION = maximizing_indicator
    RETURN_MAXIMIZATION = maximizing_return
    VOLATILITY_MINIMIZATION = volatility
    TRACKING_ERROR_MINIMIZATION = trackingError
    MEAN_VARIANCE = mean_variance
    RISK_PARITY = risk_parity
    RISK_BUDGETING = risk_budgeting

def objectiveFunction(x, function, **kwargs):
    return function(x, **kwargs)


def constraintsGeneration(order_book_ids,assetsType, union_stocks, benchmark, date, bounds, industryConstraints, industryNeutral,
                          industryDeviation, styleConstraints, styleNeutral, styleDeviation,fundTypeConstraints):

    if "*" in bounds.keys():
        bounds = {s: bounds.get("*") for s in order_book_ids}
    elif len(bounds) == 0:
        bounds = {s: (0, 1) for s in order_book_ids}

    not_weighted_stocks = set(union_stocks) - set(order_book_ids)
    bnds1 = {s: (0, 0) for s in not_weighted_stocks}
    bounds.update(bnds1)
    bnds = tuple([bounds.get(s) if s in bounds else (0, 1) for s in union_stocks])

    constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1}]
    if assetsType == "CS":
        validateConstraints(order_book_ids, bounds, date, industryConstraints, industryNeutral, styleConstraints,
                            styleNeutral)

        neutralized_industry_constraints = industry_neutralize_constraint(union_stocks, date, deviation=industryDeviation,
                                                                          industryNeutral=industryNeutral,
                                                                          benchmark=benchmark)
        customized_industry_constraints = industry_customized_constraint(union_stocks, industryConstraints, date)

        neutralized_style_constraints = style_neutralize_constraint(union_stocks, date, styleDeviation, styleNeutral,
                                                                    benchmark)
        customized_style_constraints = style_customized_constraint(union_stocks, styleConstraints, date)

        constraints.extend(neutralized_industry_constraints)
        constraints.extend(customized_industry_constraints)
        constraints.extend(neutralized_style_constraints)
        constraints.extend(customized_style_constraints)
    else:
        assert assetsType == "Fund"
        fundTypeConstraints = fund_type_constraints(order_book_ids,fundTypeConstraints)
        constraints.extend(fundTypeConstraints)
    return bnds, constraints


def covarianceEstimation(**kwargs):
    cov_estimator = kwargs.get("cov_estimator")
    daily_returns = kwargs.get("daily_returns")
    lw = LedoitWolf()
    covMat = lw.fit(daily_returns).covariance_ if cov_estimator == "shrinkage" else daily_returns.cov()
    return covMat

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

def portfolio_optimize(order_book_ids, date,indicator_series, method=OptimizeMethod.VOLATILITY_MINIMIZATION, cov_estimator="shrinkage",
                            annualized_return = None,fundTypeConstraints = None,
                           window=126, bounds=None, industryConstraints=None, styleConstraints=None,
                           benchmarkOptions=None,riskBudegtOptions=None,transactionCostOptions=None):
    """
    :param order_book_ids:股票列表
    :param date: 优化日期 如"2018-06-20"
    :param method 进行优化的目标函数
    :param cov_estimator: str 目标函数优化时的协方差矩阵
            可选参数：
                -"shrinkage" 默认值; 对收益的协方差矩阵进行收缩矩阵
                -"empirical" 计算个股收益的经验协方差矩阵
                -"multifactor" 因子协方差矩阵
    :param annualized_return: 预期年化收益，拥护
    :param window: int 计算收益率协方差矩阵时的回溯交易日数目 默认为126
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


    :return: pandas.Series优化后的个股权重 ,新股列表list(optional),停牌股list(optional)
    """
    benchmarkOptions = {} if benchmarkOptions is None else benchmarkOptions
    benchmark = benchmarkOptions.get("benchmark", "000300.XSHG")
    cov_estimator_benchmarkOptions = benchmarkOptions.get('cov_estimator')
    trackingErrorThreshold = benchmarkOptions.get("trackingErrorThreshold")
    industryNeutral, industryDeviation = benchmarkOptions.get('industryNeutral'), benchmarkOptions.get('industryDeviation')
    styleNeutral, styleDeviation = benchmarkOptions.get("styleNeutral"), benchmarkOptions.get("styleDeviation")
    industryMatching = benchmarkOptions.get("industryMatching")

    riskBudgetOptions = {} if riskBudegtOptions is None else riskBudegtOptions

    bounds = {} if bounds is None else bounds
    industryConstraints = {} if industryConstraints is None else industryConstraints
    styleConstraints = {} if styleConstraints is None else styleConstraints

    _benchmark = benchmark if method is OptimizeMethod.TRACKING_ERROR_MINIMIZATION or isinstance(trackingErrorThreshold,float) or riskBudgetOptions.get("riskMetrics") == "tracking_error" else None

    assetsType = assetsDistinguish(order_book_ids)

    if _benchmark is not None and assetsType == "Fund":
        raise Exception("[基金] 不支持 包含跟踪误差 约束 的权重优化 ")
    if annualized_return is not None and (method is OptimizeMethod.MEAN_VARIANCE or method is OptimizeMethod.RETURN_MAXIMIZATION):
        raise Exception("均值方差和预期收益最大化时需要指定 annualized_return [预期年化收益]")
    if _benchmark is not None and not isinstance(trackingErrorThreshold,float):
        raise Exception("指定 float 类型 的 trackingErrorThreshold [跟踪误差阈值]")

    order_book_ids, union_stocks, suspended_stocks, subnew_stocks = assetsListHandler(date=date,
                                                                                      order_book_ids=order_book_ids,
                                                                                      window=window,
                                                                                      benchmark=_benchmark,assetsType=assetsType)

    bnds, constraints = constraintsGeneration(order_book_ids=order_book_ids,assetsType=assetsType, union_stocks=union_stocks,
                                              benchmark=benchmark, date=date, bounds=bounds,
                                              industryConstraints=industryConstraints, industryNeutral=industryNeutral,
                                              industryDeviation=industryDeviation, styleConstraints=styleConstraints,
                                              styleNeutral=styleNeutral, styleDeviation=styleDeviation,fundTypeConstraints=fundTypeConstraints)

    start_date = pd.Timestamp(date) - np.timedelta64(window * 2, "D")
    if assetsType == 'CS':
        daily_returns = get_price(union_stocks, start_date, date, fields="close").pct_change().dropna(how='all').iloc[-window:]
    else:
        assert assetsType == "Fund"
        daily_returns = fund.get_nav(order_book_ids,start_date,date,fields="acc_net_value").pct_change().dropna(how="all").iloc[-window:]

    objective_covMat = covarianceEstimation(daily_returns=daily_returns,cov_estimator=cov_estimator) if method is OptimizeMethod.TRACKING_ERROR_MINIMIZATION else covarinaceEstimation(daily_returns=daily_returns[order_book_ids],cov_estimator=cov_estimator)

    x0 = (pd.Series(1, index=order_book_ids) / len(order_book_ids)).reindex(union_stocks).replace(np.nan, 0).values
    options = {'disp': True}

    if isinstance(trackingErrorThreshold,float):
        constraints_covMat = covarianceEstimation(daily_returns=daily_returns,
                                                  cov_estimator=cov_estimator_benchmarkOptions)
        trackingErrorOptions = {"covMat": constraints_covMat, "benchmark": benchmark, "union_stocks": union_stocks,"date":date}
        constraints.append({"type": "ineq", "fun": lambda x: -trackingError(x,**trackingErrorOptions) + trackingErrorThreshold})

    # if method is OptimizeMethod.VAR or method is OptimizeMethod.CVAR:
    #     covariance = rqdatac.barra.get_factor_covariance(date=date)
    #     specific = rqdatac.barra.get_specific_return(order_book_ids,start_date=date,end_date=date).iloc[0]
    #     factor_exposure = rqdatac.barra.get_factor_exposure(order_book_ids,start_date=date,end_date=date).xs(date, level=1)
    #     kwargs_var = {"covariance":covariance,"specific":specific,"factor_exposure":factor_exposure}

    if method is OptimizeMethod.RISK_BUDGETING:
        riskBudgetOptions.update({"assetsType":assetsType})
        riskBudgets = assetRiskAllocation(order_book_ids, riskBudgetOptions)
        constraints = []
        constraints.append({"type": "ineq", "fun": lambda x: riskBudgets.dot(np.log(x)) - 0.1})

    kwargs = {"covMat": objective_covMat, "benchmark": benchmark, "union_stocks": union_stocks, "date": date,
                  "indicator_series": indicator_series}
    kwargs.update({"riskMetrics":riskBudgetOptions.get("riskMetrics")})
    kwargs.update({"annualized_return":annualized_return})
    # kwargs.update(kwargs_var)

    # 对风险平价有效，是否带有约束条件
    if len(constraints)>1:
        kwargs.update({"with_cons":True})

    def _objectiveFunction(x):
        return objectiveFunction(x, method, **kwargs)

    res = minimize(_objectiveFunction, x0, bounds=bnds, constraints=constraints, method='SLSQP', options=options)
    optimized_weight = pd.Series(res['x'], index=union_stocks).reindex(order_book_ids)
    optimized_weight/=optimized_weight.sum()

    if industryMatching:
        # 获得未分配行业的权重与成分股权重
        unallocatedWeight, supplementStocks = benchmark_industry_matching(order_book_ids, benchmark, date)
        optimized_weight = pd.concat([optimized_weight * (1 - unallocatedWeight), supplementStocks])

    return optimized_weight, res.status, subnew_stocks, suspended_stocks


