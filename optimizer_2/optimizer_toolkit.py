# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))
from rqdatac import *
import rqdatac
import scipy.spatial as scsp
from.exception import *


def get_subnew_delisted_assets(order_book_ids,date,N,type="CS"):
    instruments_fun = fund.instruments if type == "Fund" else instruments
    all_detail_instruments = instruments_fun(order_book_ids)
    subnew_assets = [s for s in all_detail_instruments if len(get_trading_dates(s.listed_date,date))<=N] if isinstance(N,int) else []
    delisted_assets = [s for s in all_detail_instruments if (not s.de_listed_date =="0000-00-00") and pd.Timestamp(s.de_listed_date)<pd.Timestamp(date)]

    return subnew_assets,delisted_assets

def get_st_stocks(stocks,date):
    previous_date = get_previous_trading_date(date)
    st_series = is_st_stock(stocks,start_date=previous_date,end_date=date).iloc[-1]
    return st_series[st_series].index.tolist()

def get_suspended_stocks(stocks,end_date,N):
    start_date = rqdatac.trading_date_offset(end_date,-N)
    volume = get_price(stocks, start_date, end_date, fields="volume")
    suspended_day = (volume == 0).sum(axis=0)
    return suspended_day[suspended_day>0.5*N].index.tolist()
def winsorized_std(rawData, n=3):
    mean, std = rawData.mean(), rawData.std()
    return rawData.clip(mean - std * n, mean + std * n)

def assetsListHandler(filter,**kwargs):

    st_filter = filter.get("st_filter")

    benchmark = kwargs.get("benchmark")
    date = kwargs.get("date")
    order_book_ids = kwargs.get("order_book_ids")
    window = filter.get("subnew_filter")

    assetsType = kwargs.get("assetsType")

    if assetsType == "CS":

        benchmark_components = index_components(benchmark, date=date) if benchmark is not None else []
        union_stocks = sorted(set(benchmark_components).union(set(order_book_ids)))
        st_stocks = get_st_stocks(union_stocks,date) if st_filter else []

        subnew_stocks,delisted_stocks = get_subnew_delisted_assets(union_stocks, date, window)
        suspended_stocks = get_suspended_stocks(union_stocks,date,window)
        union_stocks = sorted(set(union_stocks) - set(subnew_stocks) - set(delisted_stocks) - set(st_stocks) - set(suspended_stocks))

        subnew_stocks = list(set(subnew_stocks)&set(order_book_ids))
        delisted_stocks = list(set(delisted_stocks)&set(order_book_ids))
        st_stocks = list(set(st_stocks)&set(order_book_ids))
        suspended_stocks = list(set(suspended_stocks)&set(order_book_ids))

        order_book_ids = sorted(set(union_stocks)&set(order_book_ids))

        return order_book_ids,union_stocks,subnew_stocks,delisted_stocks,suspended_stocks,st_stocks
    else:
        assert assetsType == "Fund"
        subnew_funds,delisted_funds = get_subnew_delisted_assets(order_book_ids, date, window,type="Fund")
        order_book_ids = sorted(set(order_book_ids)-set(subnew_funds)-set(delisted_funds))
        return order_book_ids,order_book_ids,subnew_funds,delisted_funds

def ensure_same_type_instruments(order_book_ids):

    instruments_func = rqdatac.instruments
    instrument = instruments_func(order_book_ids[0])
    if instrument is None:
        instrument = rqdatac.fund.instruments(order_book_ids[0])
        if instrument is not None:
            instruments_func = rqdatac.fund.instruments
        else:
            raise InvalidArgument('unknown instrument: {}'.format(order_book_ids[0]))

    all_instruments_detail = instruments_func(order_book_ids)

    instrumentsTypes = ([instrument.type for instrument in all_instruments_detail])
    cons_all_stocks = instrumentsTypes.count("CS") == len(order_book_ids)
    cons_all_funds = instrumentsTypes.count("PublicFund") == len(order_book_ids)

    if not (cons_all_funds or cons_all_stocks):
        raise InvalidArgument("传入的合约[order_book_ids]应该为统一类型")
    assetsType = "Fund" if cons_all_funds else "CS"
    return assetsType


def trackingError(x,**kwargs):

    c_m,_index_weights = kwargs.get("c_m"),kwargs.get("index_weights")
    X = x - _index_weights
    result = np.sqrt(np.dot(np.dot(X, c_m * 252),X))
    return result

def variance(x,**kwargs):
    c_m = kwargs.get("c_m")
    return np.dot(np.dot(x,c_m*252),x)

def mean_variance(x,**kwargs):

    annualized_return = kwargs.get("series")
    risk_aversion_coefficient = kwargs.get("risk_aversion_coefficient")

    if not isinstance(annualized_return,pd.Series):
        raise Exception("在均值方差优化中请指定 预期收益")
    portfolio_volatility = variance(x,**kwargs)

    return -x.dot(annualized_return) + np.multiply(risk_aversion_coefficient,portfolio_volatility)

def maximizing_series(x,**kwargs):
    series = kwargs.get("series")
    return -x.dot(series)

def maximizing_return_series(x,**kwargs):
    series = kwargs.get("series")
    return -x.dot(series)

def risk_budgeting(x,**kwargs):
    riskMetrics = kwargs.get("riskMetrics")
    if riskMetrics == "volatility":
        return np.sqrt(variance(x, **kwargs))
    else:
        assert riskMetrics == "tracking_error"
        riskBudgets = kwargs.get("riskBudgets").values
        c_m, _index_weights = kwargs.get("c_m"), kwargs.get("index_weights")
        X = x - _index_weights
        total_tracking_error = trackingError(x,**kwargs)
        contribution_to_active_risk = np.multiply(X,np.dot(c_m,X))/total_tracking_error
        c = np.vstack([contribution_to_active_risk,riskBudgets])
        res = sum(scsp.distance.pdist(c))
        return res

def risk_parity(x,**kwargs):

    c_m = kwargs.get("c_m")

    def risk_parity_with_con_obj_fun(x):
        temp1 = np.multiply(x, np.dot(c_m, x))
        temp2 = temp1[:, None]
        return np.sum(scsp.distance.pdist(temp2, "euclidean"))

    if kwargs.get("with_cons"):
        return risk_parity_with_con_obj_fun(x)
    else:
        c=15
        res =  np.dot(x, np.dot(c_m, x)) - c * sum(np.log(x))
        # print(res)
        return res


def benchmark_industry_matching(order_book_ids, benchmark, date):
    """
    返回未配置行业的权重之和/未配置每个行业的权重
    :param order_book_ids:
    :param benchmark:
    :param date:
    :return:
    """

    # 获取基准行业配置信息
    benchmark_components = index_weights(benchmark, date)
    benchmark_industry_label = shenwan_instrument_industry(list(benchmark_components.index), date=date)['index_name']
    # benchmark_industry_label = missing_industryLabel_handler(list(benchmark_components.index), date)
    benchmark_merged_df = pd.concat([benchmark_components, benchmark_industry_label], axis=1)
    benchmark_industry_allocation = benchmark_merged_df.groupby(['index_name']).sum()
    # 获取投资组合行业配置信息
    # portfolio_industry_label = missing_industryLabel_handler(order_book_ids, date)
    portfolio_industry_label = shenwan_instrument_industry(order_book_ids, date=date)['index_name']
    portfolio_industry = list(portfolio_industry_label.unique())
    missing_industry = list(set(benchmark_industry_allocation.index) - set(portfolio_industry))
    # 若投资组合均配置了基准所包含的行业，则不需要进行配齐处理
    if (len(missing_industry) == 0):
        return None, None
    else:
        matching_component = benchmark_merged_df.loc[benchmark_merged_df['index_name'].isin(missing_industry)]
    return matching_component['weight'].sum(), matching_component['weight']


def missing_industryLabel_handler(order_book_ids,date):

    return shenwan_instrument_industry(order_book_ids, date=date)['index_name']

    # industry = shenwan_instrument_industry(order_book_ids,date=date)['index_name'].reindex(order_book_ids)
    # missing_stocks = industry[industry.isnull()].index.tolist()
    #
    # if len(missing_stocks):
    #     min_date = pd.to_datetime([instruments(s).listed_date for s in missing_stocks]).min()
    #     supplemented_data = {}
    #     for i in range(1,6,1):
    #         datePoint = (min_date+np.timedelta64(i*22,"D")).date()
    #         # if datePoint
    #         industryLabels = shenwan_instrument_industry(missing_stocks,datePoint)['index_name']
    #         supplemented_data.update(industryLabels.to_dict())
    #         missing_stocks = sorted(set(missing_stocks) - set(industryLabels.index))
    #         if len(missing_stocks) == 0:
    #             break
    #     industry.loc[supplemented_data.keys()] = pd.Series(supplemented_data)
    # return industry


