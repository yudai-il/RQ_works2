import pandas as pd
import numpy as np
import scipy.stats as st
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16007))
from datetime import timedelta
from rqdatac import *
from datetime import datetime,timedelta
# rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))

from.factorICAnalysis import *

def calc_cummalativeReturns(factor_values, quantiles, bins=None, N=22):
    """
    :param factor_values: pandas.DataFrame
                                 factor_values announce_date
                000001.XSHE       0.776614    2018-04-20
                000002.XSHE      -0.943400    2018-04-26
                000008.XSHE      -0.985722    2018-04-28
    :param quantiles: int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    :param bins: int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    :param N: int
    :return: pandas.Series
    """
    _no_raise = True

    def quantile_calc(x, _quantiles, _bins, _no_raise):
        try:
            if _quantiles is not None and _bins is None:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _bins is not None and _quantiles is None:
                return pd.cut(x, _bins, labels=False) + 1
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    def _calc_Returns(stock, date):
        _price = allPrice[stock].loc[slice(date, None)].iloc[:N]
        return _price.iloc[-1] / _price.iloc[0] - 1

    factor_values["quantiles"] = quantile_calc(factor_values['factor_values'], quantiles, bins, _no_raise)
    start_date = factor_values['announce_date'].min()
    allPrice = get_price(factor_values.index.tolist(), start_date, start_date + np.timedelta64(400, "D"),
                         fields="close")
    factor_values["returns"] = [_calc_Returns(i[0], i[1]['announce_date']) for i in factor_values.iterrows()]
    return factor_values.groupby("quantiles")['returns'].mean()

def calc_periods_Returns(financial_indicator,start_year,end_year, quantiles,stocksPool,industry,YOY = False,bins=None, N=22,excludeST=True,excludeSubNew=True,subNewThres=365):
    mapping_dates = {"q1":"04-01","q2":"07-01",'q3':"10-01",'q4':"12-31"}
    all_quarters = sorted([str(i)+str(j) for j in mapping_dates.keys() for i in np.arange(start_year,end_year+1,1)])
    all_end_dates = sorted([str(i)+"-"+ j for j in list(mapping_dates.values()) for i in np.arange(start_year,end_year+1,1)])

    returns = {}

    for i,q in enumerate(all_quarters):
        print("calculating the quarter === %s"%(q))
        try:
            stocks = all_instruments(type="CS",date=all_end_dates[i]).order_book_id.tolist() if stocksPool=="A" else index_components(stocksPool,all_end_dates[i])
            industry_code = shenwan_instrument_industry(stocks,date=all_end_dates[i])['index_name']
            stocks = industry_code[industry_code == industry].index.tolist()

            if excludeST:
                stocks = filter_st_stocks(stocks,date=all_end_dates[i])
            if excludeSubNew:
                stocks = filter_subnew_stocks(stocks,all_end_dates[i],subNewThres)

            if YOY:
                _factor_values = get_yoy_quarterly_data(financial_indicator,stocks,q)
            else:
                _factor_values = get_quarterly_data(financial_indicator,stocks,q)

            returns[q] = calc_cummalativeReturns(_factor_values, quantiles, bins=None, N=N)
        except:
            pass
    return pd.DataFrame(returns)
