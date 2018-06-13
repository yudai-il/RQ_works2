import pandas as pd
import numpy as np
import scipy.stats as st
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16007))
from datetime import timedelta
from rqdatac import *
from datetime import datetime,timedelta


def get_quarterly_data(financial_indicator,stocks,quarter):
    """
    :param financial_indicator: 需要查询的因子 格式例如 fundamentals.financial_indicator_TTM.return_on_equityTTM
    :param stocks: 股票列表,list
    :param quarter: 财报的年份和季度，例"2017q3"
    :return: 假设该字段位于非资产负债表，则为该股票的财务指标变化率，否则为原始财务值 pandas.DataFrame
    """
    df = get_financials(query(financial_indicator,fundamentals.stockcode,fundamentals.announce_date).filter(fundamentals.stockcode.in_(stocks)),quarter=quarter,interval='3q')

    def _get_indicator_change(data):
        data = data.dropna().sort_index()
        # 对于非资产负债表的字段，取其变化值
        if len(data) < 3:
            # 假设这个股票财务指标有缺失
            return [np.nan, np.nan]
        else:
            # 假设当前排序是 q4,q1,q2 则 q1, q2-q1
            if data.index[-1][-2:] == "q2":
                return [data.iloc[-2], data.iloc[-1] - data.iloc[-2]]
            # 假设当前排序是 q3,q4,q1 则 q4-q3 , q1
            elif data.index[-1][-2:] == "q1":
                # 其余差分
                return [data.iloc[-2] - data.iloc[-3], data.iloc[-1]]
            else:
                return data.diff().dropna().values.tolist()

    original_factor_values = df.major_xs(quarter)
    announce_dates = pd.to_datetime(original_factor_values['announce_date'],format='%Y%m%d')

    if str(financial_indicator).startswith("StkBalaGen.total_assets"):
        original_factor_values.columns = ['factor_values','announce_date']
        original_factor_values['announce_date'] = announce_dates
        return original_factor_values
    else:
        data = df.iloc[0]
        factor_values = pd.DataFrame(pd.DataFrame(data.apply(lambda x:_get_indicator_change(x)).to_dict()).pct_change().iloc[-1]).astype(float)
        factor_values.columns = ['factor_values']
        factor_values['announce_date'] = announce_dates
        return factor_values


def calc_quarterly_imformationCoefficient(factor_values,N=22):
    """
    :param factor_values: 因子值
    :param N:滞后的交易日数目
    :return:返回该季度时期的相关系数(spearman) float
    """
    def _calc_single_returns(stock,date):
        try:
            _price = get_price(stock,date,date+np.timedelta64(40,"D"),fields='close').iloc[-N:]
            _returns = 0 if len(_price) ==0 else _price[-1]/_price[0]-1
            return _returns
        except:
            return np.nan
    returns = pd.Series([_calc_single_returns(i[0],i[1]['announce_date']) for i in factor_values.iterrows()],index=factor_values.index)
    ic = st.spearmanr(factor_values['factor_values'],returns,nan_policy='omit')[0]
    return ic
    
#--------------------------------------------The following for testing-------------------------------------------


def calc_periods_imformationCoefficient(financial_indicator,start_year,end_year,stocksPool):
    """
    :param financial_indicator: 需要查询的因子 格式例如 fundamentals.financial_indicator_TTM.return_on_equityTTM
    :param start_year: 开始年份 Integer 2014
    :param end_year: 结束年份 Integer 2018
    :param stocksPool: 指数股票池, eg. "000300.XSHG","000905.XSHG"
    :return: a series like
                    2014q1   -0.087843
                    2014q2    0.017007
                    2014q3   -0.021521
                    2014q4   -0.006395
                    2015q1   -0.059323

    """

    mapping_dates = {"q1":"03-31","q2":"06-30",'q3':"09-30",'q4':"12-31"}
    all_quarters = sorted([str(i)+str(j) for j in mapping_dates.keys() for i in np.arange(start_year,end_year+1,1)])
    all_end_dates = sorted([str(i)+"-"+ j for j in list(mapping_dates.values()) for i in np.arange(start_year,end_year+1,1)])
    ics = {}
    for i,q in enumerate(all_quarters):
        print("calculating the quarter === %s"%(q))
        try:
            _factor_values = get_quarterly_data(financial_indicator,index_components(stocksPool,all_end_dates[i]),q)
            ic = calc_quarterly_imformationCoefficient(_factor_values,N=22)
        except:
            ic = np.nan
        ics[q] = ic
    return pd.Series(ics)

# --------------duPont Analysis-------------------

financial_indicator_roe = fundamentals.financial_indicator.du_return_on_equity
financial_indicator_NPM = fundamentals.financial_indicator.du_profit_margin
financial_indicator_EM = fundamentals.financial_indicator.du_equity_multiplier
financial_indicator_AU = fundamentals.financial_indicator.du_asset_turnover_ratio

ics_roe_CSI500 = calc_periods_imformationCoefficient(financial_indicator_roe,2014,2018,"000905.XSHG")
ics_NPM_CSI500 = calc_periods_imformationCoefficient(financial_indicator_NPM,2014,2018,"000905.XSHG")
ics_AU_CSI500 = calc_periods_imformationCoefficient(financial_indicator_AU,2014,2018,"000905.XSHG")
ics_EM_CSI500 = calc_periods_imformationCoefficient(financial_indicator_EM,2014,2018,"000905.XSHG")

ics_roe_CSI300 = calc_periods_imformationCoefficient(financial_indicator_roe,2014,2018,"000300.XSHG")
ics_NPM_CSI300 = calc_periods_imformationCoefficient(financial_indicator_NPM,2014,2018,"000300.XSHG")
ics_AU_CSI300 = calc_periods_imformationCoefficient(financial_indicator_AU,2014,2018,"000300.XSHG")
ics_EM_CSI300 = calc_periods_imformationCoefficient(financial_indicator_EM,2014,2018,"000300.XSHG")


net_profit_margin = pd.concat([ics_NPM_CSI500,ics_NPM_CSI300],axis=1)
equity_multiplier = pd.concat([ics_EM_CSI500,ics_EM_CSI300],axis=1)
assets_turnover = pd.concat([ics_AU_CSI500,ics_AU_CSI300],axis=1)
return_on_equity = pd.concat([ics_roe_CSI500,ics_roe_CSI300],axis=1)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2,sharex=True,figsize=(8,6))
ax = ax.flatten()
net_profit_margin.plot(kind='bar',ax=ax[0])
ax[0].set_title("销售净利率")
equity_multiplier.plot(kind='bar',ax=ax[1])
ax[1].set_title("权益乘数")
assets_turnover.plot(kind='bar',ax=ax[2])
ax[2].set_title("总资产周转率")
return_on_equity.plot(kind='bar',ax=ax[3])
ax[3].set_title("净资产收益率")
