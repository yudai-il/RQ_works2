#Notes During Practising

此目录下是一些实习期的代码

- 因子方面包括
    -
        1、显式因子计算
        2、简易行业分析
        3、跟踪误差计算
        4、因子相关信息系数和收益率分析
        
- 优化器
    - 
<font color=#DC143C size=5 face="黑体">NOTES</font>

    optimizer_1 和optimizer_2 的区别：
    
    > optimizer_1为开发期的迭代版本
    
    > optimizer_2是综合各板块功能的版本，增强了可读性
    
<hr/>
-


> 一、目标函数

        1、波动率最小化
        2、跟踪误差最小化
        3、预期收益最大化
        4、均值方差（效用函数）
        5、指标最大化
        6、风险平价
        7、* 风险预算
> 二、约束条件

        1、资产头寸约束
        2、主动头寸约束
        3、行业约束
            a/自定义约束
            b/偏离约束
        4、风格约束
            a/自定义约束
            b/偏离约束
        5、跟踪误差约束
> 三、交易费用 
        
    ! ATTENTIONS: 此部分暂不支持包含在优化器中

        交易成本的组成部分
            1、佣金
            2、买卖价差（未考虑，需要五档行情数据）
            3、市场冲击成本
            4、机会成本（难以测量）
            
>四、其他

        1、对ST、次新股、停牌股的处理
        2、行业配齐

>五、参考文献

        [1]Xi Bai,Katya Scheinberg,Reha Tutuncu. Least-squares approach to risk parity in portfolio selection[J]. Quantitative Finance,2016,16(3).
        [2]Jorion P. Portfolio Optimization with Tracking-Error Constraints[J]. Financial Analysts Journal, 2003, 59(5):70-82.
        [3]Benjamin Bruder, Thierry Roncalli. Managing Risk Exposures Using the Risk Budgeting Approach[J]. Mpra Paper, 2012.

        
        
    
    
    


