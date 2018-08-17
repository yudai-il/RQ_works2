import pandas as pd
import numpy as np
import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))
from rqdatac import *

return_risk = pd.read_pickle("optimizer_v_develop/testSamples/imformations.pkl")


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
fig,ax = plt.subplots()
return_risk["returns"].plot(ax=ax)
ax.set_xticklabels(labels=return_risk.index,rotation=90)
plt.legend(['预期年化收益'],loc='upper center',bbox_to_anchor=(0.5,1),framealpha=0)
ax = ax.twinx()
return_risk['risk'].plot(ax=ax,c='g')
plt.legend(['初始风险贡献比例'],loc='upper center',bbox_to_anchor=(0.5,0.9),framealpha=0)
plt.tick_params(labelsize=15)

plt.grid(True)
plt.show()


weights = pd.read_pickle("optimizer_v_develop/testSamples/weights.pkl")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
fig,ax = plt.subplots()
weights[:3.5].plot(ax=ax)
plt.ylabel("个股头寸",fontsize=18)
plt.xlabel("收益风险比",fontsize=18)
plt.tick_params(labelsize=15)
plt.legend(fontsize=12,loc="upper left")
plt.grid(True)
plt.show()


