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
    :param quarter: 财报的年份和季度，例"2107q3"
    :return: 假设该字段位于非资产负债表，则为该股票的财务指标变化率，否则为原始财务值
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

    if str(financial_indicator).startswith("fundamentals.balance_sheet"):
        return df.major_xs(quarter)
    else:
        data = df.iloc[0]
        factor_values = pd.DataFrame(pd.DataFrame(data.apply(lambda x:_get_indicator_change(x)).to_dict()).pct_change().iloc[-1]).astype(float)
        factor_values['announce_date'] = pd.to_datetime(df.major_xs(quarter)['announce_date'],format="%Y%m%d")
        return factor_values


def calc_quarterly_imformationCoefficient(financial_indicator,stocks,quarter,N=22):
    """
    :param financial_indicator:
    :param stocks:股票列表
    :param quarter:财报的年份和季度，例"2107q3"
    :param N:滞后的交易日数目
    :return:返回该季度时期的相关系数(spearman)
    """
    df = get_quarterly_data(financial_indicator, stocks, quarter).dropna()

    def _calc_single_returns(stock,date):
        try:
            _price = get_price(stock,date,date+np.timedelta64(40,"D"),fields='close').iloc[-N:]
            _returns = 0 if len(_price) ==0 else _price[-1]/_price[0]-1
            return _returns
        except:
            return np.nan
    returns = pd.Series([_calc_single_returns(i[0],i[1].iloc[1]) for i in df.iterrows()],index=df.index)
    ic = st.spearmanr(df.iloc[:,0],returns,nan_policy='omit')[0]
    return ic
    
#--------------------------------------------The following for testing-------------------------------------------

financial_indicator = fundamentals.income_statement.net_profit
# stocks
# all_a_stocks = all_instruments(type="CS").order_book_id.tolist()

mapping_dates = {"q1":"03-31","q2":"06-30",'q3':"09-30",'q4':"12-31"}
all_quarters = [str(i)+str(j) for j in mapping_dates.keys() for i in np.arange(2014,2018,1)]
all_end_dates = [str(i)+"-"+ j for j in list(mapping_dates.values()) for i in np.arange(2014,2018,1)]
all_end_dates.sort()
all_quarters.sort()
ics = {}
for i,q in enumerate(all_quarters):
    print(i)
    ic = calc_quarterly_imformationCoefficient(financial_indicator,index_components("000300.XSHG",all_end_dates[i]),q,N=22)
    ics[i] = ic

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
pd.Series(ics).plot(kind='bar',ax=ax)
ax.set_xticklabels(all_quarters)
