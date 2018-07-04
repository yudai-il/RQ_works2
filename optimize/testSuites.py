import pandas as pd
import numpy as np
import rqdatac
rqdatac.init('rice', 'rice', ('192.168.10.64', 16009))
#from ..factorOptimization import optimizer,optimizer_toolkit
#from factor_analysis_multifactor.factorOptimization import optimizer,optimizer_toolkit
from functools import reduce

def get_daily_return_stock_weight(portfolio,rebalancing_dates,end_date,cash_return):

    period_division = rebalancing_dates + [end_date]

    portfolio_daily_return = pd.Series()

    portfolio_weight = pd.DataFrame()

    for period in range(0, len(period_division) - 1):
        start_weight = portfolio[period_division[period]]

        if 'cash' in start_weight.index:
            processed_weight = start_weight.drop('cash')
        else:
            processed_weight = start_weight

        # 计算个股权重的时候，用个股的实际收盘价

        start_stock_price = rqdatac.get_price(processed_weight.index.tolist(), period_division[period], period_division[period], fields='close',adjust_type='none').loc[period_division[period]].T

        if period_division[period + 1] == end_date:
            daily_price = rqdatac.get_price(processed_weight.index.tolist(),rqdatac.get_next_trading_date(period_division[period]).strftime("%Y-%m-%d"), end_date,fields='close', adjust_type='none')

        else:
            daily_price = rqdatac.get_price(processed_weight.index.tolist(),rqdatac.get_next_trading_date(period_division[period]).strftime("%Y-%m-%d"), rqdatac.get_previous_trading_date(period_division[period+1]).strftime("%Y-%m-%d"),fields='close', adjust_type='none')

        # 每天更新权重：start_weight * price/start_date_price,加入cash weight若原策略不含有cash，cash_weight=0
        stock_daily_weight = ((processed_weight / start_stock_price) * daily_price).T

        stock_daily_weight.loc['cash'] = np.repeat([start_weight.loc['cash'] if 'cash' in start_weight.index else 0],len(stock_daily_weight.columns))

        start_weight.name = pd.Timestamp(rqdatac.get_previous_trading_date(stock_daily_weight.columns[0]))

        stock_daily_weight = pd.concat([start_weight,stock_daily_weight],axis=1)

        # 归一化权重

        normalized_stock_weight = (stock_daily_weight / stock_daily_weight.sum())

        # 计算个股收益率的时候，使用前复权收盘价
        stock_daily_return = rqdatac.get_price(processed_weight.index.tolist(), period_division[period], period_division[period+1], fields='close',
                                               adjust_type='pre').pct_change()[1:]

        # 根据用户选择计算投资组合cash_return
        if cash_return is None:
            cash_daily_return = pd.Series(index=stock_daily_return.index, data=0)
        elif isinstance(cash_return, float):
            cash_daily_return = pd.Series(index=stock_daily_return.index, data=(1 + cash_return) ** (1 / 252) - 1)
        else:

            compounded_risk_free_return = rqdatac.get_yield_curve(period_division[period], period_division[period+1],
                                                                  tenor=str.upper(cash_return[-2:]))[str.upper(cash_return[-2:])]
            cash_daily_return = pd.Series(index=stock_daily_return.index,
                                          data=(1 + compounded_risk_free_return.loc[stock_daily_return.index]) ** (1 / 252) - 1)

        stock_daily_return['cash'] = cash_daily_return

        portfolio_daily_return_period = pd.Series(index=stock_daily_return.index)
        for trading_date in stock_daily_return.index.tolist():

            last_trading_date = rqdatac.get_previous_trading_date(trading_date)

            portfolio_daily_return_period.loc[trading_date] = (normalized_stock_weight[last_trading_date] * stock_daily_return.loc[trading_date]).sum()

        portfolio_daily_return = pd.concat([portfolio_daily_return,portfolio_daily_return_period])

        portfolio_weight = pd.concat([portfolio_weight,normalized_stock_weight],axis=1).replace(np.nan,0)

    return portfolio_daily_return,portfolio_weight.T


# start_date = "2014-01-01"
# end_date = "2018-06-30"
#
# dates = pd.DataFrame(rqdatac.get_trading_dates(start_date,end_date))
# dates["ym"] = dates[0].astype(str).str[:7]
dates = dates.groupby("ym").first()[0].values

# date=monthlyDates[0]

# 1/2/3
def portfolio(dates,size=10):
    all_results = {}
    for date in dates:
        marketCap = rqdatac.get_factor(rqdatac.all_instruments(type="CS",date=date).order_book_id.tolist(),"a_share_market_val",date=date)
        marketCap.sort_values(inplace=True)
        marketCapIndex = pd.Series(marketCap.index)
        stocksLists = [marketCapIndex.truncate(i,i+size).values.tolist() for i in range(len(marketCapIndex))[::size]]
        results = list(map(lambda x:optimizer.volatility_minimization(x,date)[0],stocksLists[:]))
        all_results[date] = results
    return all_results

# 4
def portfolio_4(dates):
    all_results = {}

    for date in dates:
        all_stocks = rqdatac.all_instruments(type="CS", date=date).order_book_id.tolist()
        industryData = rqdatac.shenwan_instrument_industry(all_stocks,date=date)['index_name']
        industry_stocks_mapping = industryData.groupby(industryData).apply(lambda x:x.index.tolist()).to_dict()
        results = {industry:optimizer.volatility_minimization(industry_stocks_mapping.get(industry),date)[0] for industry in industry_stocks_mapping.keys()}
        all_results[date] = results

#   5,6,7
def portfolio_5(dates,size):
    all_results1 = {}
    all_results2 = {}

    def _get_max_marketCap(x,marketCap):
        mc = marketCap.loc[x].sort_values().index.tolist()
        return mc[:size],mc[-size:]

    for date in dates:
        all_stocks = rqdatac.all_instruments(type="CS", date=date).order_book_id.tolist()
        marketCap = rqdatac.get_factor(all_stocks,"a_share_market_val",date=date)
        industryData = rqdatac.shenwan_instrument_industry(all_stocks,date=date)['index_name']
        industry_stocks_mapping = industryData.groupby(industryData).apply(lambda x:x.index.tolist()).to_dict()
        res = {x:_get_max_marketCap(industry_stocks_mapping.get(x), marketCap) for x in industry_stocks_mapping.keys()}
        min_size_stocks = sorted(reduce(lambda x,y:set(x)|set(y),[res.get(key)[0] for key in res.keys()]))
        max_size_stocks = sorted(reduce(lambda x,y:set(x)|set(y),[res.get(key)[1] for key in res.keys()]))
        results1 = optimizer.volatility_minimization(min_size_stocks)[0]
        results2 = optimizer.volatility_minimization(max_size_stocks)[0]
        all_results1[date] = results1
        all_results2[date] = results2

def portfolio_8(dates,size):
    all_results = {}

    def _get_max_marketCap(x,marketCap):
        mc = marketCap.loc[x].sort_values().index.tolist()
        return mc[:size],mc[-size:]

    for date in dates:
        all_stocks = rqdatac.all_instruments(type="CS", date=date).order_book_id.tolist()
        marketCap = rqdatac.get_factor(all_stocks,"a_share_market_val",date=date)
        industryData = rqdatac.shenwan_instrument_industry(all_stocks,date=date)['index_name']
        industry_stocks_mapping = industryData.groupby(industryData).apply(lambda x:x.index.tolist()).to_dict()
        res = {x:_get_max_marketCap(industry_stocks_mapping.get(x), marketCap) for x in industry_stocks_mapping.keys()}
        min_size_stocks = reduce(lambda x,y:set(x)|set(y),[res.get(key)[0] for key in res.keys()])
        max_size_stocks = reduce(lambda x,y:set(x)|set(y),[res.get(key)[1] for key in res.keys()])
        stocks = sorted(set(min_size_stocks)|set(max_size_stocks))
        results = optimizer.volatility_minimization(stocks)[0]
        all_results[date] = results



















# portfolios.groupby(portfolios).apply(lambda x: optimizer.volatility_minimization(x,date))




