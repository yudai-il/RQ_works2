import pandas as pd
import numpy as np
import rqdatac
rqdatac.init('rice', 'rice', ('192.168.10.64', 16009))
# from ..factorOptimization import optimizer,optimizer_toolkit
from factor_analysis_multifactor.factorOptimization import optimizer,optimizer_toolkit
from functools import reduce


"""
计算设置：
从2014年1月1日至2018年6月30日，月频率调仓，计算日收益率序列

测试组合：
	第一类组合：全市场股票按市值排序，划分为N个股票数目为10个的组合
	第二类组合：全市场股票按市值排序，划分为N个股票数目为50个的组合
	第三类组合：全市场股票按市值排序，划分N个股票数目为100个的组合
	第四类组合：申万一级行业分类的28个行业作为股票组合
	第五类组合：买入申万一级行业分类的28个行业中：（1）各行业中市值最大的股票，共计28只股票；(2) 各行业中市值最小的股票，共计28只股票
	第六类组合：买入申万一级行业分类的28个行业中：（1）各行业中市值前二的股票，共计56只股票；(2) 各行业中市值后二的股票，共计56只股票
	第七类组合：买入申万一级行业分类的28个行业中：（1）各行业中市值前五股票，共计135只股票；(2) 各行业中市值后五股票，共计135只股票
	第八类组合：买入申万一级行业分类的28个行业中：（1）各行业中市值前五和市值后五的股票，共计270只股票 
"""

# Noticification
# the first trading day for each month
# storage the stocksList collection initially for reusing in various optimization
#

# plan 1 ~ 3
def portfolio(dates,size=10):
    stocksCollection = {}
    for date in dates:
        marketCap = rqdatac.get_factor(rqdatac.all_instruments(type="CS",date=date).order_book_id.tolist(),"a_share_market_val",date=date)
        marketCap.sort_values(ascending=False,inplace=True)
        marketCapIndex = pd.Series(marketCap.index)
        stocksLists = [marketCapIndex.truncate(i,i+size-1).values.tolist() for i in range(len(marketCapIndex))[::size]]
        stocksCollection[date] = stocksLists
    return stocksCollection

# plan 4
def portfolio_4(dates):
    stocksCollection = {}

    for date in dates:
        all_stocks = rqdatac.all_instruments(type="CS", date=date).order_book_id.tolist()
        industryData = rqdatac.shenwan_instrument_industry(all_stocks,date=date)['index_name']
        industry_stocks_mapping = industryData.groupby(industryData).apply(lambda x:x.index.tolist()).to_dict()
        stocksCollection[date] = industry_stocks_mapping
    return stocksCollection

# plan 5 ~ 7
def portfolio_5(dates,size):
    stocksCollection_a = {}
    stocksCollection_b = {}

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

        stocksCollection_a[date] = min_size_stocks
        stocksCollection_b[date] = max_size_stocks
    return stocksCollection_a,stocksCollection_b

def portfolio_8(dates,size):
    stocksCollection = {}

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
        stocksCollection[date] = stocks
    return stocksCollection


