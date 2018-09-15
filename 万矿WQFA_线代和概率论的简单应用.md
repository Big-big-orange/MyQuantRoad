## 线代和概率论的简单应用

课程链接：https://www.windquant.com/qntcloud/college







* 本小节自己学到的小技巧

* 哑变量

* 期货合约换月

* 在Dataframe的某一列中找出包含某一字符串的数据，组成新的dataframe

  futuresUpSHFE = futuresUp[futuresUp['SEC_NAME'].str.contains('SHFE')]



### 线性代数

#### 线性回归模型简介

```python
# 矩阵求逆
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.linalg.inv(A)
```

假设被解释变量$y$与多个解释变量$X_1,X_2,...,X_p$之间具有线性关系，则记为

$$
\large y=\beta_0+\beta_1X_1+\beta_2X_2+...+\beta_pX_p+\mu
$$
称之为多元线性回归模型，其中$\beta_0,\beta_1,...,\beta_p$为$p+1$个未知参数，$\mu$为随机扰动项，表示除$X_1,X_2,...,X_p$以外影响$y$的所有非观测因素，并假设$\mu\sim N(0,\sigma^2)$。

假设有$n$组观测数据$y_i,X_{1i},X_{2i},...,X_{pi},i=1,2,...,n$，则存在线性方程组

$$
\large
\begin{equation}
\left\{                        
\begin{aligned}
y_1=\beta_0+\beta_1X_{11}+\beta_2X_{21}+...+\beta_pX_{p1}+\mu_1\\
y_2=\beta_0+\beta_1X_{12}+\beta_2X_{22}+...+\beta_pX_{p2}+\mu_2\\
\vdots\\
y_n=\beta_0+\beta_1X_{1n}+\beta_2X_{2n}+...+\beta_pX_{pn}+\mu_n\\
\end{aligned}
\right.
\end{equation}
$$
若记$Y=(y_1,y_2,...,y_n)^\tau$，
$
X=\left[
 \begin{matrix}
   1&X_{11} & X_{21}&...&X_{p1} \\
   1&X_{12} & X_{22}&...&X_{p2} \\
   \vdots & \vdots&\ddots\\
   1&X_{1n} & X_{2n}&...&X_{pn} \\
  \end{matrix}
  \right]
$，$\beta=(\beta_1,\beta_2,...,\beta_n)^\tau$，$\mu=(\mu_1,\mu_2,...,\mu_n)^\tau$，

则线性方程组可表示为
$$
\large
Y=X\beta+\mu
$$
通过参数估计方法，可以得到未知参数$\beta$的估计值。

$$
\large
\hat{\beta}=(X^\tau X)^{-1}X^\tau Y
$$


#### 线性回归模型求解模拟

statsmodels是Python中一个强大的统计分析包，包含了回归分析、时间序列分析、假设检验等等的功能，当需要在Python中进行回归分析时，就可以导入statsmodels。

statsmodels.regression.linear_model里有回归函数statsmodels.OLS，它的输入参数有(endog, exog, missing, hasconst)。一般只考虑前两个输入，其中，endog是回归中的因变量$Y$，是一个$n$维的向量；exog是回归中的自变量$X_1,X_2,...,X_P$，由于statsmodels.OLS不会假设回归模型有常数项，所以我们应该假设模型是

$$
\large
y_t=\beta_0X_{0t}+\beta_1X_{1t}+\beta_2X_{2t}+...+\beta_pX_{pt}+\mu,t=1,2,...,n
$$
其中，对所有$t=1,2,...,n$，令$X_{0t}=1$。因此，exog的输入是一个$n\times (p+1)$的向量。

statsmodels.OLS的输出结果是statsmodels.regression.linear_model.OLS类，并没有进行任何运算。在OLS的模型之上调用拟合函数 fit()，才进行回归运算，并且得到statsmodels.regression.linear_model.RegressionResultsWrapper，它包含了这组数据进行回归拟合的结果摘要。调用params可以查看计算出的回归系数$\beta_0,\beta_1,...,\beta_p$。



一般而言，有连续取值的变量叫做连续变量，它的取值可以是任何的实数，或者是某一区间里的任意实数，比如股价、市价等。但有些变量的取值不是连续的，只能有有限个取值，比如交易状态、股票所属行业等，这些变量是离散变量。在回归分析中，需要把离散变量转化为哑变量。

如果想表达有$d$种取值的离散变量，则对应的哑变量的取值是一个$d$元组，其中有一个元素为1，其余元素为0。元素为1处就是变量对应的类型。比如离散变量的取值为a、b、c、d，那么类别a对应的哑变量为(1,0,0,0)，类别b对应的哑变量是(0,1,0,0)，类别c对应的哑变量是(0,0,1,0)，类别d对应的哑变量是(0,0,0,1)。假设a、b、c、d四种情况对应的系数为$\beta_1$、$\beta_2$、$\beta_3$、$\beta_4$，设$(z_1,z_2,z_3,z_4)$是一个取值对应的哑变量，那么

$$
\large
\beta_1z_1+\beta_2z_2+\beta_3z_3+\beta_4z_4
$$
可以直接得出相应的系数。可以理解为，离散变量的取值本身只是分类，无法构成任何线性关系，但是若映射到高元的0,1点上，便可以用线性关系表达，从而进行回归。

statsmodels里有一个函数categorical()可以直接把类别${0,1,2,...,d-1}$转化为对应的元组。statsmodels.categorical()的输入有(data, col, dictnames, drop) 四个，中间两个输入可以不管。其中，data是一个$n\times 1$的向量，记录每一个样本的分类变量取值；drop是一个Bool值，表示是否在输出中丢掉样本变量值。

设有因变量$Y$，自变量存在连续变量$X_1$和离散变量$X_2$，其中$X_2$的取值范围为{a,b,c}，a类的系数为1，b类的系数为3，c类的系数为8，也就是将$X_2$转化为哑变量$(Z_1,Z_2,Z_3)$，$Z_i$取值为0、1；另外，常数项为10，$X_1$的系数为1，线性方程为

$$
\large
Y=10+X_1+Z_1+3Z_2+8Z_3+\mu
$$


可以用常规的方法进行 OLS 回归。

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# (1) 数据模拟
nsample = 100                                      # 确定样本量
x1 = np.linspace(0, 20, nsample)                   # 确定连续变量样本

x2 = np.zeros(nsample, int)                        # 定义离散自变量
x2[0:int(nsample/5)] = 0                           # 前20个样本取值为0
x2[int(nsample/5):int(3*nsample/5)] = 1            # 中间20个变量取值为1
x2[int(3*nsample/5):] = 2                          # 最后10个变量取值为2

dummy = sm.categorical(x2, drop=True)              # 离散变量转变为哑变量

X = sm.add_constant(np.column_stack((x1, dummy)))  # 连续变量和哑变量合并，并添加常数项

beta = [10, 1, 1, 3, 8]                            # 确定模型系数真值
mu = np.random.normal(size=nsample)                # 确定模型随机扰动值

y = np.dot(X, beta) + mu                           # 生成解释变量值

result = sm.OLS(y,X).fit()
print(result.params)

# (2) 画图
plt.figure(figsize = (12,4))
plt.axis((0, 20, 10, 40))                    # 设置坐标轴区间
plt.grid(True)
plt.plot(x1, y, 'o', label='data')          # 绘出原始数据
plt.plot(x1, result.fittedvalues, 'r--.',label='OLS')  # 绘出拟合数据
plt.legend(loc='best')                     # 添加图注释
```

![1536759765819](Photo\1536759765819.png)

这里要指出，哑变量是和其他自变量并行的影响因素，也就是说，哑变量和原先的X1X1同时影响了回归的结果。初学者往往会误解这一点，认为哑变量是一个选择变量：也就是说，上图中给出的回归结果，是在只做了一次回归的情况下完成的，而不是分成３段进行３次回归。哑变量的取值藏在其他的三个维度中。可以理解成：上图其实是将高元的回归结果映射到平面上之后得到的图。







#### 回归法多因子选股

回归法就是利用历史数据去拟合而得到拟合系数作为每个因子的权重。

举例：

1. 观察主体为全部A股成分股中的食品饮料行业股，因变量为月收益率，自变量为上月末因子值，因子取为市盈率、市净率、流通市值、换手率。
2. 对数据进行标准化, 剔除极端值。
3. 上月末因子值与月收益率拟合, 确定本月因子权重, 对月末股票打分并于下月初买入。

```python
from WindPy import *
w.start()

from datetime import datetime
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm


# 获取模型训练数据

beginDate = "20170101"     # 回测起始日期                                        
endDate = "20180601"       # 回测截止日期

beginDateBefore1 = datetime.strftime(w.tdaysoffset(-1,beginDate).Data[0][0],'%Y%m%d')
monthEndTradeDate=w.tdays(beginDateBefore1, endDate, "Period=M").Data[0][:-1]        #月末交易日

buyInfo = []
for t in range(len(monthEndTradeDate)-1):
    
    endMonthUp = datetime.strftime(monthEndTradeDate[t],'%Y%m%d')
    endMonth = datetime.strftime(monthEndTradeDate[t+1],'%Y%m%d')
    
    # 全A股食品饮料行业个股
    aStock = ','.join(w.wset("sectorconstituent","date="+endMonth+";sectorid=a001010100000000").Data[1])
    aStockIndustry = w.wss(aStock,"industry2","industryType=1;industryStandard=1;tradeDate="+endMonth) 
    foodDrinkCode = []
    for i in range(len(aStockIndustry.Data[0])):
        if aStockIndustry.Data[0][i]=="食品饮料":
            foodDrinkCode.append(aStockIndustry.Codes[i])
    
    # 获取市盈率、市净率、流通市值、换手率、交易状态、涨跌幅数据
    factorDate1 = w.wss(foodDrinkCode, 'pe_ttm,pb_lyr,mkt_freeshares,turn,trade_status,maxupordown',"tradeDate="+endMonthUp)

    # 计算月涨跌幅
    chg = w.wss(foodDrinkCode, "pct_chg_per","startDate="+endMonthUp+";endDate="+endMonth).Data[0]
    
    # 准备回归数据
    stockData = factorDate1.Data
    stockData.insert(4,chg)
    stockDataDF=pd.DataFrame(stockData,index=["pe_ttm", "pb_lyr", "mkt_freeshares", "turn", "chg", "trade_status", "maxupordown"],columns=factorDate1.Codes).T
    stockDataDF = stockDataDF.dropna()
    
    # 数据标准化
    stockDataDF["chg"] = preprocessing.scale(stockDataDF["chg"]) # preprocessing.scale标准化函数
    stockDataDF["pe_ttm"] = preprocessing.scale(stockDataDF["pe_ttm"])
    stockDataDF["pb_lyr"] = preprocessing.scale(stockDataDF["pb_lyr"])
    stockDataDF["mkt_freeshares"] = preprocessing.scale(stockDataDF["mkt_freeshares"])
    stockDataDF["turn"] = preprocessing.scale(stockDataDF["turn"])
    
    # 数据去极值
    for field in stockDataDF.columns[0:5]:
        mean = np.mean(stockDataDF[field])
        std = np.std(stockDataDF[field])
        upValue = mean + 2*std
        downValue = mean - 2*std        
        for windcode in stockDataDF.index:
            if stockDataDF[field][windcode] > upValue:
                stockDataDF[field][windcode] = upValue
            elif stockDataDF[field][windcode] < downValue:
                stockDataDF[field][windcode] = downValue
    
    X = sm.add_constant(np.column_stack((stockDataDF['pe_ttm'].tolist(), stockDataDF['pb_lyr'].tolist(),stockDataDF['mkt_freeshares'].tolist(),stockDataDF['turn'].tolist()))) 
    y = stockDataDF['chg'].tolist()
    # 这里算出来的result就是回归法得到的因子权重。
    result = sm.OLS(y,X).fit()

    factorDate2 = w.wss(foodDrinkCode, 'pe_ttm,pb_lyr,mkt_freeshares,turn,trade_status,maxupordown',"tradeDate="+endMonth)
    factorDate2DF=pd.DataFrame(factorDate2.Data,index=["pe_ttm","pb_lyr","mkt_freeshares","turn","trade_status","maxupordown"],columns=factorDate2.Codes).T
    factorDate2DF = factorDate2DF[factorDate2DF["trade_status"]=="交易"][factorDate2DF["maxupordown"]==0].dropna()
    
    # 标准化因子数据
    factorDate2DF["pe_ttm"] = preprocessing.scale(factorDate2DF["pe_ttm"])
    factorDate2DF["pb_lyr"] = preprocessing.scale(factorDate2DF["pb_lyr"])
    factorDate2DF["mkt_freeshares"] = preprocessing.scale(factorDate2DF["mkt_freeshares"])
    factorDate2DF["turn"] = preprocessing.scale(factorDate2DF["turn"])
    
    for field in factorDate2DF.columns[0:4]:
        mean = np.mean(factorDate2DF[field])
        std = np.std(factorDate2DF[field])
        upValue = mean + 2*std
        downValue = mean - 2*std        
        for code in factorDate2DF.index:
            if factorDate2DF[field][code] > upValue:
                factorDate2DF[field][code] = upValue
            elif factorDate2DF[field][code] < downValue:
                factorDate2DF[field][code] = downValue
    
    windCode = list(factorDate2DF.index.values)
    
    rawmat = np.mat(factorDate2DF)
    scoreBook = {} 
    selectStock = []
    
    weights = result.params[1:5]
    
    for i in range(len(windCode)):
        secID = windCode[i]
        x = rawmat[i,0:4]
        score = (np.array(np.dot(x,weights)))
        scoreBook.update({secID:score[0][0]}) 
    top5 = sorted(scoreBook.items(), key=lambda scoreBook: scoreBook[1])[-6:-1]  
    
    for i in top5:
        selectStock.append(i[0])    
    
    buyInfo.append(selectStock)   
    
    
    
    
from WindAlgo import * #引入回测框架

df2 = pd.DataFrame()

def initialize(context):#定义初始化函数
    context.capital = 100000 #回测的初始资金
    context.start_date ="20170201" #回测开始时间
    context.end_date = "20180601" #回测结束时间
    context.benchmark = '000300.SH'  #设置回测基准为沪深300  
    context.securities = ['000001.SZ']      #回测标的 因为后面会改变股票池 所以这里可以任意设定一个 
    
    context.index = 0
    
def handle_data(bar_datetime, context, bar_data):
    pass

def my_schedule(bar_datetime, context, bar_data):                  # 注意：schedule函数里不能加入新的参数
    bar_datetime_str = bar_datetime.strftime('%Y-%m-%d')           # 调仓时间
    
    buyList = buyInfo[context.index]
    # 食品饮料股纳入股票池
    wa.change_securities(buyList) 
    
    sellList = list(wa.query_position().get_field('code'))     
    for code in sellList:
        if code not in buyList:
            volumn = wa.query_position()[code]['volume']    #找到每个code的持仓量 
            res = wa.order(code,volumn,'sell',price='close', volume_check=False)  
            # 卖出上月初买入的股票, 本月计划买入的不卖出
    
    for code in buyList:
        if code not in sellList:
            wa.order(code,100,'buy',price='close', volume_check=False)
            
    context.index = context.index + 1

wa = BackTest(init_func = initialize, handle_data_func=handle_data)   #实例化回测对象
wa.schedule(my_schedule, "m", 0)         #   m表示在每个月执行一次策略 0表示偏移  表示月初第一个交易日往后0天
res = wa.run(show_progress=True)          #调用run()函数开始回测,show_progress可用于指定是否显示回测净值曲线图
nav_df = wa.summary('nav')                  #获取回测结果  回测周期内每一天的组合净值    
```

![1536761768534](Photo\1536761768534.png)





### 概率论



#### 二项分布

二项分布通常用来描述成功/失败情况，在投资中，许多决策都会出现两种结果。如果只做一次成功/失败试验，我们称之为伯努利试验，伯努利试验有两个可能的结果:

$$
\large p(1) = p(Y = 1) = p
$$

$$
\large p(0) = p(Y = 0)= 1-p
$$

注：记1表示成功，相应$p$代表成功的概率.

二项分布则是进行$n$次伯努利试验, 其中成功的次数$X$为随机变量, 并且每次伯努利试验相互独立. 随机变量$X$整体分布由$n$与$p$两个参数决定, 可记为$X\sim B(n,p)$.

为了计算二项分布随机变量的概率分布函数，我们需要从总试验中选出成功的个数，这是一个组合问题（从一个集合中选出元素的所有可能，不考虑次序），从$n$个元素中选出$x$个的排列定义如下：


$$
\large \begin{pmatrix}
n\\
x
\end{pmatrix}=\frac{n!}{(n-x)!x!}
$$

使用排列公式可以很容易地得到二项分布的概率分布函数：

$$
\large p(x) = P(X = x) =\frac{n!}{(n-x)!x!}p^x(1-p)^{n-x}
$$

注：表示从n个元素中选出x个标识为成功，其余为失败



#### 正态分布

正态分布是统计学中非常普遍且重要的一种分布，许多统计学方法与检验（金融分析）都是以数据满足正态分布为前提假设的. 在一些量化交易策略中，正态分布也为我们提供了许多便捷，例如许多配对交易策略，就是以证券对间的价差变化服从正态分布为基础的。

正态分布有两个参数，均值$\mu$和方差$\sigma^2$，$X$服从正态分布可以写为：

$$
\large X\sim N(\mu,\sigma^2)
$$

现代组合理论中，通常都假设证券收益服从正态分布，而且正态分布还具有一个很关键的特征，多个服从正态分布的随机变量的线性组合仍然服从正态分布，这就为证券组合及其后的分析提供了极大的便利。

正态分布的概率密度函数如下：

$$
\large f(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\{-\frac{(x-\mu)^2}{2\sigma^2}\},-\infty<x<+\infty
$$



#### 置信区间

由于样本均值和总体的均值是不一样的. 一般来说，我们想知道总体均值, 但普查难以实施且成本高, 通常我们只能计算样本均值. 然后我们就需要用样本的均值去估计总体的均值. 为了刻画样本均值估算总体均值的精度, 我们使用置信区间来决定衡量.

如果需要估算全校所有学生的身高，可以测量一下10个学生的身高，然后用这个身高均值去估算总体的均值。



但仅获得样本的均值并没有什么用, 并不知道它跟总体均值的联系程度. 为了确定这个相关程度, 我们可以求一下样本的方差值. 方差高说明样本的不稳定性和不确定性较高.

计算标准误差包括假设样本无偏的，并且数据是正态且独立的。如果这些条件被违反了，得出的标准误差将是错误的。

标准误差的计算公式是
$$
\large SE = \frac{\sigma}{\sqrt n}
$$
假设数据是正态分布的，然后我们就可以使用标准误差来计算置信区间. 首先，我们先设置需要的置信度，比如95%，然后来看总体的95%包含多少标准差。结果是总体的95%位于标准正态分布的 -1.96 和 1.96 之间。当样本的数量足够大的时候（一般门槛为大于30），中心极限定理能够适用，正态性也可以被安全保障；如果样本的规模比较小，一个较好的方式是使用适当的特定自由度的t分布..

**注意：使用中心极限定理的时候要小心，金融中的很多数据集基本是非正态的，所以未经考虑地使用这个理论或者不注意其中的区别是不科学的。**



将总体的95%进行图表可视化：

```python
# 设置 x 轴
x = np.linspace(-5,5,100)
# 标准正态分布
y = stats.norm.pdf(x,0,1)

plt.figure(figsize = (11,7))
plt.grid(True)
plt.title("Normal distribution")
plt.xlabel('$\sigma$')
plt.ylabel('Normal PDF');

plt.plot(x,y)

# 绘制边界
plt.vlines(-1.96, 0, 0.5, colors='r', linestyles='dashed')
plt.vlines(1.96, 0, 0.5, colors='r', linestyles='dashed')

# 填充区域
fill_x = np.linspace(-1.96, 1.96, 500)
fill_y = stats.norm.pdf(fill_x, 0, 1)
plt.fill_between(fill_x, fill_y)

plt.show()
```

![1536762491272](Photo\1536762491272.png)、







#### 概率分布与置信区间在统计套利中的应用



思路：在期货的交易中，我们认为期货的价差服从正态分布，然后利用分布的置信区间确定开平仓信号，实现统计套利。

![1536762815453](Photo\1536762815453.png)

配对交易的准备步骤，利用期货合约涨跌幅的相关性做统计套利。并得到了(HC.SHF-RB.SHF)的收盘价差diff, 计算收盘价差的近20日均值$\mu$和标准差$\sigma$, 绘制价差diff、均线$\mu$、上轨线$\mu+2\sigma$和下轨线$\mu-2\sigma$, 交易信号设置如下

1. 价差上穿上轨线时, 做多RB.SHF, 做空HC.SHF（后续行情即走跌）
2. 价差下穿下轨线时，做空RB.SHF,做多HC.SHF
3. 价差触及均线时, 清仓RB.SHF和HC.SHF



```python
from WindPy import *
w.start()

from datetime import *
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


# 研究数据起止日期，结束日期时前一个交易日，起始日期是一年前
endDate = datetime.strftime(w.tdaysoffset(-1,datetime.strftime(datetime.now(),"%Y%m%d")).Data[0][0],"%Y%m%d")
beginDate = datetime.strftime(datetime.now()-timedelta(365),"%Y%m%d")


# 1.期货品种筛选
# 获取期货日行情(主力以持仓量计算)获取期货日行情(主力以持仓量计算)
error_code,futuresDF = w.wset("sectorconstituent", "date="+endDate+";sectorId=1000010084000000",usedf=True)

# 删去白银和黄金，因为他们属于现货
futuresDF = futuresDF.drop(['2','6'],axis=0) # 恰好白银和黄金位于第2行和第6行
futuresDF.head(10)
code= ','.join(futuresDF['wind_code'].values)

#按成交量排序
error_code,futuresDF =w.wss(code, "sec_name,exch_eng,volume", "tradeDate="+endDate+";priceAdj=1;cycle=1",usedf=True)

# 打印出主力合约成交量在 10000 以上的期货合约
futuresUp =futuresDF[futuresDF.VOLUME > 1000 ]
print( )
print('主力合约成交量在 10000 以上的有：' ,len(futuresUp),'个')
print( )
print('分别是：',list(futuresUp['SEC_NAME']))


#选取上期所的期货合约作为研究对象
futuresUp[futuresUp['SEC_NAME'].str.contains('SHFE')]


futuresUpSHFE = futuresUp[futuresUp['SEC_NAME'].str.contains('SHFE')]
print( )
print('上期所合约数为',len(futuresUpSHFE),'个')

codeSHFE = list(futuresUpSHFE.index) # 得到合约名的列表

# 获得在回测日期内各期货的涨跌幅
dfSHFE = []
for code in codeSHFE:  
    errorcode,dataSHFEtemp = w.wsd(code, "windcode,pct_chg", beginDate, endDate, "Fill=Previous",usedf=True)
    dfSHFE.append(dataSHFEtemp)
dfSHFE = pd.concat(dfSHFE)
# 获得回测区间内的数据
dfSHFE = dfSHFE.dropna()
dataStr = dfSHFE.index.strftime('%Y-%m-%d')
dfSHFE.set_index(dataStr,inplace=True)
dfSHFE.index.name = 'tradeDate'

# 将所有的数据拼接起来，组成一个dataframe
SHFECode = list(set(dfSHFE['WINDCODE'])) #去除了重复对象  
dfSHFEResult= DataFrame() 
for i in SHFECode:
    dfSHFEtemp = DataFrame(dfSHFE[dfSHFE['WINDCODE']==i]['PCT_CHG'])
    dfSHFEtemp.columns = [i]
    dfSHFEResult = pd.concat([dfSHFEResult,dfSHFEtemp],axis=1)
# 计算品种之间的相关性   
data_SHFE = pd.DataFrame(dfSHFEResult,dtype=np.float)
shang_corr = data_SHFE.corr(method='spearman')
shang_corr    
```

![1536840252157](Photo\1536840252157.png)

```python
# 从中取出相关性最高的两个期货品种
tempCorrDF = shang_corr
for indexs in tempCorrDF.index:
    for columns in tempCorrDF.columns:
        if indexs == columns:
            tempCorrDF.loc[indexs].loc[columns] = 0

[(indexs,columns) for indexs in tempCorrDF.index for columns in tempCorrDF.columns if(tempCorrDF.loc[indexs].loc[columns] ==tempCorrDF.max().max())]
```

```python
# 画出实际的上轨下轨和均线
from datetime import *

tempBeginDate = datetime.strftime(w.tdaysoffset(-21,beginDate,"").Data[0][0],"%Y%m%d")
tempEndDate = endDate

errorCode,stockData = w.wsd("HC.SHF,RB.SHF", "close", tempBeginDate, tempEndDate, "Fill=Previous",usedf=True)
stockDataDiff = stockData["HC.SHF"]-stockData["RB.SHF"]

muList = []
sigmaList = []
upLine = []
downLine = []
for i in range(len(stockDataDiff)):
    if i >= 20:
        muList.append(np.mean(stockDataDiff[(i-20):(i-1)])) # 每天近二十日价差的均值
        sigmaList.append(np.std(stockDataDiff[i-20:(i-1)]))
        upLine.append(muList[i-20] + 2*sigmaList[i-20])
        downLine.append(muList[i-20] - 2*sigmaList[i-20])

plt.figure(figsize=(25,7))
plt.grid(True)
plt.title("HC.SHF与RB.SHF的涨跌幅差值序列")
plt.xlabel("Date")
plt.ylabel("Diff")

plt.plot(muList)
plt.plot(upLine)
plt.plot(downLine)
plt.plot(list(stockDataDiff[20:]))
plt.show()
```

![1536840746889](Photo\1536840746889.png)

```
# 回测

from WindAlgo import * #引入回测框架

#定义初始化函数
def initialize(context):#定义初始化函数
    context.capital = 100000 #回测的初始资金
    context.securities = ["HC.SHF","RB.SHF"]  #回测标的
    context.start_date = beginDate #回测开始时间
    context.end_date = endDate     #回测结束时间
    context.period = 'd' #策略运行周期, 'd' 代表日
    context.benchmark = "HC.SHF"

    # 研究中得到的股票残差开仓线，为箱体图的二倍标准差内
    context.mean =  muList
    context.up_line = upLine
    context.down_line = downLine    
    context.index = 1
    
    context.flag = 0  #合约换约标志
    
#定义策略函数
def handle_data(bar_datetime, context, bar_data):
    
    today = bar_datetime.strftime('%Y-%m-%d')  
    yesterday = w.tdaysoffset(-1, today).Data[0][0].strftime('%Y-%m-%d')  
    tomorrow = w.tdaysoffset(1, today).Data[0][0].strftime('%Y-%m-%d') 

    stockData1 = w.wsd(",".join(context.securities), "close",yesterday, today)    
    difftemp1 = stockData1.Data[0][1] - stockData1.Data[1][1]  #当日价差 
    difftemp2 = stockData1.Data[0][0] - stockData1.Data[1][0]  #昨日价差

    stockData2 = w.wsd(",".join(context.securities),"trade_hiscode",today,tomorrow)
    securities1 = [stockData2.Data[0][0], stockData2.Data[1][0]]   #当日主力合约
    securities2 = [stockData2.Data[0][1], stockData2.Data[1][1]]   #明日主力合约        
    
    currentPosition=bkt.query_position()
    
    # 合约换月前一天清仓
    if securities1[0] != securities2[0] or securities1[1] != securities2[1]: 
        if context.securities[0] in currentPosition.get_field('code'):
            holdnum1 = currentPosition[context.securities[0]]['volume']
            holdnum2 = currentPosition[context.securities[1]]['volume'] 
            if currentPosition[context.securities[0]]['side'] == 'long':            
                bkt.order(context.securities[0], holdnum1, "sell", "price=close", volume_check=False)        
                bkt.order(context.securities[1], holdnum2, "cover","price=close", volume_check=False)
            elif currentPosition[context.securities[0]]['side'] == 'short':           
                bkt.order(context.securities[0], holdnum1, "cover", "price=close", volume_check=False)       
                bkt.order(context.securities[1], holdnum2, "sell", "price=close", volume_check=False)
            context.flag = 1
    else:
        # 合约换月当天不交易
        if context.flag == 1:      
            context.flag = 0
        else:        
            # 前日价差大于均线, 当日价差上穿上轨线, 做多HC.SHF, 做空RB.SHF
            if difftemp1 > context.up_line[context.index] and difftemp2 < context.up_line[context.index-1] and difftemp2 > context.mean[context.index-1] and context.securities[1] not in currentPosition.get_field('code'):
                bkt.order(context.securities[1], 10, "buy", price='close', volume_check=False)
                bkt.order(context.securities[0], 10, "short", price='close', volume_check=False)
            # 当日价差大于下轨线, 当日价差下穿均线, 清仓
            elif difftemp1 < context.mean[context.index] and difftemp2 > context.mean[context.index-1] and difftemp1 > context.down_line[context.index]:        
                if context.securities[1] in currentPosition.get_field('code'):
                    bkt.order(context.securities[1], currentPosition[context.securities[1]]['volume'], "sell", price='close', volume_check=False)
                    bkt.order(context.securities[0], currentPosition[context.securities[0]]['volume'], "cover", price='close', volume_check=False)
            
            # 前日价差小于均线, 当日价差下穿下轨线, 做空HC.SHF, 做多RB.SHF
            if difftemp1 < context.down_line[context.index] and difftemp2 > context.down_line[context.index-1] and difftemp2 < context.mean[context.index-1] and context.securities[0] not in currentPosition.get_field('code'):
                bkt.order(context.securities[0], 10, "buy", price='close', volume_check=False)
                bkt.order(context.securities[1], 10, "short", price='close', volume_check=False)
            # 当日价差小于上轨线， 当日价差上穿均线, 清仓
            elif difftemp1 > context.mean[context.index] and difftemp2 < context.mean[context.index-1] and difftemp1 < context.up_line[context.index]:
                if context.securities[1] in currentPosition.get_field('code'):
                    bkt.order(context.securities[0], currentPosition[context.securities[0]]['volume'], "sell", price='close', volume_check=False)
                    bkt.order(context.securities[1], currentPosition[context.securities[1]]['volume'], "cover", price='close', volume_check=False)
            
            # 当日连续下穿均线和下轨线
            if difftemp1 < context.down_line[context.index] and difftemp2 > context.mean[context.index-1]:
        
                if context.securities[1] in currentPosition.get_field('code'):
                    bkt.order(context.securities[1], currentPosition[context.securities[1]]['volume'], "sell", price='close', volume_check=False)
                    bkt.order(context.securities[0], currentPosition[context.securities[0]]['volume'], "cover", price='close', volume_check=False)
            
                    bkt.order(context.securities[0], 10, "buy", price='close', volume_check=False)
                    bkt.order(context.securities[1], 10, "short", price='close', volume_check=False)

                else:
                    bkt.order(context.securities[0], 10, "buy", price='close', volume_check=False)
                    bkt.order(context.securities[1], 10, "short", price='close', volume_check=False)
            # 当日连续上穿均线和上轨线
            elif difftemp1 > context.up_line[context.index] and difftemp2 < context.mean[context.index-1]:

                if context.securities[1] in currentPosition.get_field('code'):
                    bkt.order(context.securities[0], currentPosition[context.securities[0]]['volume'], "sell", price='close', volume_check=False)
                    bkt.order(context.securities[1], currentPosition[context.securities[1]]['volume'], "cover", price='close', volume_check=False)
            
                    bkt.order(context.securities[1], 10, "buy", price='close', volume_check=False)
                    bkt.order(context.securities[0], 10, "short", price='close', volume_check=False)
            
                else:
                    bkt.order(context.securities[1], 10, "buy", price='close', volume_check=False)
                    bkt.order(context.securities[0], 10, "short", price='close', volume_check=False)

    context.index = context.index + 1 

bkt = BackTest(init_func = initialize, handle_data_func=handle_data) #实例化回测对象
res = bkt.run(show_progress=True) #调用run()函数开始回测,show_progress可用于指定是否显示回测净值曲线图
nav_df=bkt.summary('nav') # 获取回测结果 
```

