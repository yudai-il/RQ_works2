from.optimizer import *

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
        item_names = ['费用', '费率', '份额']
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
        merged_data['自定义费用'] = customizedCost.loc[all_assets].replace(np.nan,0)
        merged_data['总费用'] = merged_data['费用']+merged_data['自定义费用']
        merged_data['费率'] = merged_data['总费用']/pd.concat([long_position,short_position])


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

        merged_data = pd.DataFrame({"交易金额":positions,"印花税费率":stamp_duty_cost,"佣金费率":commission,"市场冲击成本":impact_cost,"出清时间（天）":clearing_time,"买卖方向":sides,"自定义费用":customizedCost.loc[all_assets].replace(np.nan,0)})
        merged_data['总交易费用'] = commission+impact_cost+merged_data['自定义费用']
    if output:
        return merged_data
    else:
        if assetType == "CS":
            return merged_data['总交易费用']*(252/holdingPeriod)
        else:
            return merged_data['费率']*(252/holdingPeriod)

def calc_cummulativeReturn_analysis(positions,date,N,assetType):
    order_book_ids = positions.index.tolist()
    start_date = pd.Timestamp(date)-np.timedelta64(N,"D")
    price_data = fund.get_nav(order_book_ids, start_date=start_date, end_date=date,
                              fields="adjusted_net_value") if assetType == "Fund" else get_price(order_book_ids,
                                                                                                 start_date, date,
                                                                                                 fields="close")
    return price_data.pct_change().dropna(how="all").add(1).prod()-1


def calc_risk_indicator(positions,date,N,assetType,cov_estimator):
    positions = positions[~(positions.index == "cash")]
    order_book_id = positions.index.tolist()
    start_date = pd.Timestamp(date) - np.timedelta64(N, "D")
    price_data = fund.get_nav(order_book_id, start_date=start_date, end_date=date,
                              fields="adjusted_net_value") if assetType == "Fund" else get_price(order_book_id,
                                                                                                 start_date, date,
                                                                                                 fields="close")
    positions = positions*price_data.iloc[-1]
    positions/=positions.sum()

    daily_returns = price_data.pct_change().dropna(how="all")
    kwargs = {"daily_returns":daily_returns,"cov_estimator":cov_estimator}
    covMat = covarianceEstimation(daily_returns,cov_estimator)
    kwargs["covMat"] = covMat
    total_volatility = volatility(positions.values,**kwargs)

    def calc_nominator(x,covMat):
        return x.dot(covMat)*252

    nominators = calc_nominator(positions,covMat)

    MRC = nominators/total_volatility
    CTR = np.multiply(positions,MRC)

    return {"边际风险贡献":MRC,"风险贡献":CTR}


def tradeAnalysis(initialCapital,currentPositions,plannedPositions,covEstimator,commission=True,subscriptionRedemption=True,stampDuty=True,
                  marketImpact=True,commissionRatio=0.0008,subRedRatio=None,marketImpactRatio=1,customizedCost=None,output=True):

    if currentPositions is not None:
        start_date = list(currentPositions.keys())[0]
        currentPositions = currentPositions.get(start_date)
        currentPositions = currentPositions[~("cash" == currentPositions.index)]
    else:
        if initialCapital is None:
            raise Exception("初始金额和初始仓位中必须指定一个")

    end_date = list(plannedPositions.keys())[0]
    plannedPositions = plannedPositions.get(end_date)

    equity_value = plannedPositions[~("cash" == plannedPositions.index)]
    weights = equity_value/equity_value.sum()
    order_book_ids = weights.index.tolist()
    x = weights.values

    transactionOptions = {"initialCapital": initialCapital, "currentPositions": currentPositions,
                           "commission": commission,
                          "subscriptionRedemption": subscriptionRedemption, "stampDuty": stampDuty,
                          "marketImpact": marketImpact, "commissionRatio": commissionRatio,
                          "subRedRatio": subRedRatio, "marketImpactRatio": marketImpactRatio,
                          "customizedCost": customizedCost, "output": True
                          }


    all_assets = sorted(set(order_book_ids)|set(currentPositions.index)) if currentPositions is not None else sorted(order_book_ids)
    assetType = assetsDistinguish(all_assets)

    transactionAnalysis = calcTransactionCost(order_book_ids,x,end_date,assetType,transactionOptions)

    before_risk = calc_risk_indicator(currentPositions,start_date,252,assetType,covEstimator) if currentPositions is not None else pd.DataFrame(columns=["边际风险贡献","风险贡献"])
    after_risk = calc_risk_indicator(plannedPositions,end_date,252,assetType,covEstimator)

    before_risk = pd.DataFrame(before_risk).reindex(all_assets).replace(np.nan,0)
    after_risk = pd.DataFrame(after_risk).reindex(all_assets).replace(np.nan,0)

    delta = after_risk-before_risk
    delta.columns = ['边际风险贡献变化','风险贡献变化']
    after_risk.rename(columns = {"边际风险贡献":"计划持仓边际风险","风险贡献":"计划持仓风险贡献"},inplace=True)
    before_risk.rename(columns = {"边际风险贡献":"当前持仓边际风险","风险贡献":"当前持仓风险贡献"},inplace=True)

    res1 = calc_cummulativeReturn_analysis(plannedPositions, end_date, 7, assetType)
    res2 = calc_cummulativeReturn_analysis(plannedPositions, end_date, 30, assetType)
    res3 = calc_cummulativeReturn_analysis(plannedPositions, end_date, 90, assetType)

    returnAnalysis = pd.DataFrame({"最近一周累积收益":res1,"最近一个月累积收益":res2,"最近三个月累积收益":res3})
    riskAnalysis = pd.concat([before_risk,after_risk,delta],axis=1)


    x = transactionAnalysis[['佣金费率','印花税费率','市场冲击成本','自定义费用','总交易费用']]

    conclusion_cost = transactionAnalysis['交易金额'].dot(x)/(transactionAnalysis['交易金额'].sum())
    conclusion_cost = conclusion_cost.apply(lambda x:"{}%".format(np.round(x*100,4)))

    conclusion_risk = riskAnalysis[['当前持仓风险贡献', '计划持仓风险贡献']].sum()

    output_df = pd.concat([returnAnalysis,riskAnalysis,transactionAnalysis],axis=1)


    output_df['名称'] = pd.Series([instruments(s).symbol for s in all_assets],index=all_assets)
    columns_items = ['名称','买卖方向','最近一个月累积收益', '最近一周累积收益', '最近三个月累积收益', '当前持仓边际风险', '当前持仓风险贡献',
       '计划持仓边际风险', '计划持仓风险贡献', '边际风险贡献变化', '风险贡献变化', '佣金费率',
       '出清时间（天）', '印花税费率', '市场冲击成本', '自定义费用', '总交易费用']

    output_df = output_df[columns_items]
    output_df.sort_values(by='买卖方向',inplace=True)

    format_columns = ['最近一个月累积收益', '最近一周累积收益', '最近三个月累积收益','佣金费率','印花税费率', '市场冲击成本', '自定义费用', '总交易费用']
    output_df[format_columns] = output_df[format_columns].apply(lambda x:x.apply(lambda x:"{}%".format(np.round(x*100,4)))).replace("nan%",np.nan)

    output_df.loc['汇总'] = pd.concat([conclusion_cost,conclusion_risk]).reindex(output_df.columns.tolist()).replace(np.nan,'--')

    return output_df
    # return {"收益分析":returnAnalysis,"风险分析":riskAnalysis,"交易费用分析":transactionAnalysis}



