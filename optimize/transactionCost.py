def calcTransactionCost(date,order_book_ids,x,assetType,transactionOptions):
    initialCapital = transactionOptions.get("initialCapital")
    currentPositions = transactionOptions.get("currentPositions")
    holdingPeriod = transactionOptions.get("holdingPeriod")
    commission = transactionOptions.get("commission")
    subscriptionRedemption = transactionOptions.get("subscriptionRedemption")
    stampDuty = transactionOptions.get("stampDuty")
    marketImpact = transactionOptions.get("marketImpact")
    commissionRatio = transactionOptions.get("commissionRatio",0.0008)
    subRedRatio = transactionOptions.get("subRedRatio")
    marketImpactRatio = transactionOptions.get("marketImpactRatio",1)
    customizedCost = transactionOptions.get("customizedCost")
    cashPosition = transactionOptions.get("cashPosition")

    commissionRatio = commissionRatio if commission else 0
    marketImpactRatio = marketImpactRatio if marketImpact else 0
    subRedRatio = subRedRatio if subscriptionRedemption else {}

    # 获得当前持仓的最新价格
    cash = currentPositions['cash'] if "cash" in currentPositions.index else 0
    holding_assets = currentPositions[~(currentPositions.index == "cash")].index.tolist() if currentPositions is not None else []
    all_assets = sorted(set(holding_assets) | set(order_book_ids))
    latest_price = fund.get_nav(all_assets,start_date=date,end_date=date,fields="unit_net_value").iloc[0] if assetType == "Fund" else get_price(all_assets,start_date=date,end_date=date,fields="close",adjust_type="none").iloc[0]

    # 获得当前持有的权益
    currentCapital = (latest_price*currentPositions[~("cash" == currentPositions.index)]).replace(np.nan,0) if currentPositions is not None else latest_price*0
    #  总权益
    total_equity = initialCapital if currentPositions is None else currentCapital.sum()+cash
    # 用户非现金权益
    remaining_equity = total_equity*(1-cashPosition)
    # 获得权益的变化
    x = pd.Series(x,index=order_book_ids).reindex(all_assets).replace(np.nan,0)

    equity_delta = (x*remaining_equity) - currentCapital
    buyingAssets = equity_delta[equity_delta>0]
    sellingAssets = -equity_delta[equity_delta<0]

    # 对于基金
    if assetType == "Fund":

        itemsNames = ['费用', '费率', '份额']
        subscription_ratios = pd.Series({s.order_book_id:subRedRatio.get(s.fund_type,(0,0))[0]for s in fund.instruments(buyingAssets.index.tolist())})
        redemption_ratios = pd.Series({s.order_book_id:subRedRatio.get(s.fund_type,(0,0))[1] for s in fund.instruments(sellingAssets.index.tolist())})

        netSubCapital = buyingAssets/(1+subscription_ratios)
        subscriptionFees = buyingAssets - netSubCapital
        subscription_costs = pd.concat([subscriptionFees,subscriptionFees/buyingAssets,netSubCapital/latest_price.loc[buyingAssets.index]],axis=1)
        subscription_costs.columns = itemsNames
        subscription_costs['交易方向'] = "申购"

        redemption_costs = pd.concat([redemption_ratios*sellingAssets,redemption_ratios,sellingAssets/latest_price.loc[sellingAssets.index]],axis=1)
        redemption_costs.columns = itemsNames
        redemption_costs['交易方向'] = '赎回'

        merged_data = pd.concat([subscription_costs,redemption_costs])
    else:
        assert assetType == "CS"
        # 对于股票
        data = get_price(all_assets,pd.Timestamp(date)-np.timedelta64(400,"D"),date,fields=["close",'volume'],adjust_type="pre").iloc[-253:]
        close_price = data['close']
        volume = data['volume']
        assetsDailyVolatility = close_price.pct_change().dropna(how="all").std()
        assetsVolume = volume.iloc[-5:].mean()

        capital = pd.concat([sellingAssets,buyingAssets])
        capital.name = "交易金额"
        merged_data = pd.DataFrame(capital)
        sides = pd.Series(["SELL" if s in sellingAssets.index  else "BUY" for s in all_assets],index=all_assets)
        amounts = capital/latest_price
        clearing_time = amounts / assetsVolume
        impactCost =  marketImpactRatio * assetsDailyVolatility * (clearing_time)**(1/2)
        stampDutyCost = pd.Series([0.001 if stampDuty and s in sellingAssets.index  else 0 for s in all_assets],index=all_assets)

        commission = pd.Series([commissionRatio]*len(all_assets),index=all_assets)

        merged_data = pd.DataFrame({"印花税率费":stampDutyCost,"佣金费率":commission,"市场冲击成本":impactCost,"出清时间":clearing_time,"买卖方向":sides,"自定义费用":customizedCost.loc[all_assets].replace(np.nan,0)})
        merged_data['总交易费用'] = commission+impactCost+merged_data['自定义费用']

    return merged_data
