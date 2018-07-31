import pandas as pd
import numpy as np
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16007))
from rqdatac import *

def get_shenwan_data(date,level="SW1"):
    """
    :param date: string "2017-02-01"
    :param level: string SW1/SW2/SW3
    :return: dict 获得关于申万行业 代码和名称的映射
    {'801010.INDX': '农林牧渔',
     '801020.INDX': '采掘',
     '801030.INDX': '化工',
     '801040.INDX': '钢铁',
     '801050.INDX': '有色金属',
     '801080.INDX': '电子'}
    """
    all_a_stks = all_instruments(type='CS', date=date).order_book_id.tolist()
    shenwanIndustries = shenwan_instrument_industry(all_a_stks, date=date)
    first_level = shenwanIndustries[['index_code','index_name']].set_index("index_code").drop_duplicates()['index_name'].to_dict()

    second_level = shenwanIndustries[['second_index_code','second_index_name']].set_index('second_index_code').drop_duplicates()['second_index_name'].to_dict()

    third_level = shenwanIndustries[['third_index_code','third_index_name']].set_index('third_index_code').drop_duplicates()['third_index_name'].to_dict()

    mapping = {"SW1": first_level, "SW2": second_level, "SW3": third_level}
    return mapping.get(level)


def industry_returns_analysis(start_date,end_date,level='SW1'):
    """
    :param start_date:开始日期
    :param end_date: 结束日期
    :param level: 申万行业级别 SW1/SW2/SW3
    :return: pandas.series

    """
    shenwan_data = get_shenwan_data(end_date, level=level)
    industry_list = list(shenwan_data.keys())
    _price = get_price(industry_list,start_date,end_date,frequency='1d',fields='close')
    monthly_rtns = _price.resample('M').apply(lambda x:x.iloc[-1]/x.iloc[0]-1)
    over_zero = monthly_rtns[monthly_rtns>0].count(axis=1)
    all_count = monthly_rtns.count(axis=1)
    return over_zero/all_count

def industry_volatility_analysis(start_date,end_date,level="SW1"):
    """
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param level: 申万行业级别 SW1/SW2/SW3
    :return: pandas.DataFrame
    """
    shenwan_data = get_shenwan_data(end_date, level=level)
    industry_list = list(shenwan_data.keys())
    _price_chg = get_price_change_rate(industry_list,start_date,end_date)
    monthly_vol = _price_chg.resample("M").apply(lambda x:x.std())
    return monthly_vol

def industry_with_Index_returns_analysis(start_date,end_date,level="SW1",targetIndex="000001.XSHG"):
    """
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param level: 申万行业级别 SW1/SW2/SW3
    :param targetIndex: 收益率高于目标指数的占比
    :return: pandas.Series
    """
    shenwan_data = get_shenwan_data(end_date, level=level)
    industry_list = list(shenwan_data.keys())
    _price = get_price(industry_list,start_date,end_date,frequency='1d',fields='close')
    indexPrice = get_price(targetIndex,start_date,end_date,frequency='1d',fields="close")
    monthly_rtns = _price.resample('M').apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    indexChg = indexPrice.resample("M").apply(lambda x:x.iloc[-1]/x.iloc[0]-1)
    overIndex = monthly_rtns.apply(lambda x:x>indexChg)
    overIndex = overIndex.astype(int).sum(axis=1)
    all_count = monthly_rtns.count(axis=1)
    return overIndex/all_count

def industry_turnover(start_date,end_date,level="SW1"):
    """
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param level: 申万行业级别 SW1/SW2/SW3
    :return: pandas.DataFrame 获得每个行业的月度平均换手率
    """
    all_a_stks = all_instruments(type="CS",date=end_date).order_book_id.tolist()
    MAP = {"SW1":"index_code","SW2":"second_index_code","SW3":"third_index_code"}
    _turnover_rate = get_turnover_rate(all_a_stks,start_date,end_date,fields='today').T
    _turnover_rate['industry'] = shenwan_instrument_industry(all_a_stks,date=end_date)[MAP.get(level)]
    turnover_mean = _turnover_rate.groupby('industry').apply(lambda x:x.mean())
    return turnover_mean
