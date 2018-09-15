## 因子分析

课程链接：https://www.windquant.com/qntcloud/college

### 

- 本小节自己学到的小技巧

   

  万矿的单因子分析中的自带函数，可能以后得自己实现，可以借鉴其返回参数形式，非常简洁明了。

  **处理pd数据，整体保留小数点后四位**

  ```python
  ret_ana.return_stats.applymap(lambda x: round(x,4) if not isinstance(x, tuple) else (round(x[0],4), round(x[1],4)))
  ```

  ix实际上是取行数据，可以取代loc和iloc函数




### 因子选股的概念

因子就是**指标或者特征**, 如PE、PB、5日均线等。因子选股模型就是通过分析各个因子与股票表现（收益率）之间的关系而建立的一套量化选股的体系。

![](Photo\万矿1.JPG)



那么，怎么判断多个因子是否有效呢？

![万矿2](Photo\万矿2.JPG)

多因子选股的流程

![1536375938553](C:\Users\bigrao\AppData\Local\Temp\1536375938553.png)





### 数据的预处理

#### 1、数据的获取

	在考虑这个问题之前，我们需要确定选股的频率（日度还是月度？），以及在哪里选股（股票池，如沪深300），选择哪些因子。
	
	假设我们以沪深300指数成分股为股票池，每月调仓。因子选择'TECH_AD20', 'FA_ROENP_TTM', 'FA_NPGR_TTM'

所以我们需要以下数据（每个月最后一个交易日）：

* 沪深300成分股的数据

* 所有成分股的因子数据

* 所有成分股的收益率

* 所有成分股的市值数据

* 所有成分股的所属行业

* 策略收益比较基准的数据


#### **2、因子数据分类**

一般用作多因子选股模型的因子可以分为两大类：技术分析类因子和基本面分析类（主要基于财报数据）因子。

* 技术类因子主要有行情数据加工而来，又可分为趋势动量类、反转类及波动类等因子；

* 基本面类因子主要有盈利、成长、估值、收益、资本结构及宏观等；

当然因子的分析与研究不限于以上两类因子，投资者可以根据自身的研究，挖掘更多有效的因子。所以多因子的研究主要就是挖掘有效的因子，而怎么评价因子的有效性，是有一套通用且成熟的方法论。



在万矿中获取数据的函数如下

```python
from WindPy import * #api
from datetime import datetime, timedelta
from scipy import stats, optimize
from WindCharts import *
import pandas as pd
import WindAlpha as wa

w.start(show_welcome=False)


# 指标列表
# ind_code =   ['pe_ttm','ps_ttm'] # 因子
# raw_inds_ret = wa.prepare_raw_data('000300.SH',ind_code, '2018-01-01', '2018-04-30')

#为了在万矿平台上节省流量，最好将数据存为csv文件，数据结构为pd.DataFrame, index为MultiIndex, 保存为数据文件
raw_inds_ret.to_csv('data/ind_data.csv')

# raw_inds_ret #万矿的返回数据包括因子数据，市场的基本数据和下期的收益率

raw_inds_ret = pd.read_csv('data/ind_data.csv', index_col=[0,1]) # 指定行列索引，第一列为时间，第二列为股票
raw_inds_ret.index.names = ['dates','codes']

raw_inds_ret

## level 0 的index为时间 ，get_level_values这里是返回index的值
raw_inds_ret.index.get_level_values(0).unique()
```

![1536389053821](Photo\1536389053821.png)

##### 

+ 大家在做多因子模型的时候，在数据预处理方面，经常会遇到 **`去极值、标准化`** 这两种方法。


+ 在遇到一些停牌、数据异常等问题时，我们需要对**`缺失值`**进行一个填充。


+ 在处理金融数据时，还有一个头疼的问题就是**`“数据对齐”`**，两个相关的时间序列的索引可能没有很好地对齐，或者两个DataFrame对象可能含有不匹配的列或行等等

  ##### 

#### 3、去极值

去极值的主要目的是为了使因子数据在一个合理的范围之内，而不会因为某些异常值对因子的整体分布造成影响，去极值的方法一般有三种：

* 平均绝对离差法（MAD）：序列x中，每个元素与x序列均值相减后差的绝对值再求平均值就是MAD（Mean Absolute Deviation)
  $$
  \large MAD = \frac{\sum_{i=1}^n |X_i - \mu|}{n} 
  $$

  ```python
  def MAD(series, k = 1.4826): #比例因子 1.4826，为了达到渐进正态一致性。
      ss = series.copy()
      median = np.median(series, axis=0)
      diff = np.abs(series - median)
      mad = np.median(diff)
      std = k * mad
      #采用正负1.96个标准差
      # 查标准正态分布表，分位数1.96处对应的阴影部分的面积(概率)为0.0250。由于正态分布是两边对称的，
      # 所以1.96的置信区间对应的概率就是1-0.025*2=0.95。
      Max = median + 1.96*std
      Min = median - 1.96*std
      # clip这个函数将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
      return np.clip(series, Max, Min)
  
  
  error_code,data_Winsorzation1 = w.wset("sectorconstituent", "date=2018-04-25;sectorId=1000000090000000",usedf=True) 
  #沪深300指数成分股
  
  code1 = ','.join(data_Winsorzation1['wind_code'].values) #获取指数成分股代码
  error_code,Winsorzationpemad =w.wss(code1, "pe", "tradeDate=2018-04-25;ruleType=10",usedf=True)
  Winsorzationpemad['PE_MAD'] = MAD(Winsorzationpemad['PE'])
  
  fig, ax = plt.subplots(figsize=(14, 5))
  Winsorzationpemad['PE'].plot(color="r", ax=ax)
  MAD(Winsorzationpemad['PE']).plot(color="b", ax=ax)
  plt.xlabel("stock code")
  plt.legend(["PE", "PE_MAD"])
  ax.grid()
  ```

  ![1536391829951](Photo\1536391829951.png)

* 标准差法（Std）

  加入数据是类正态分布的时，可以使用n倍标准差,一般设置置信区间为95%，或者取±3倍标准差之外的数据定义为异常值。这个范围可自己定义。

  ```python
  error_code,data_Winsorzation2 = w.wset("sectorconstituent", "date=2018-04-25;sectorId=1000000090000000",usedf=True)
  #沪深300指数成分股
  
  code2 = ','.join(data_Winsorzation2['wind_code'].values) #获取指数成分股代码
  error_code,Winsorzationpestd =w.wss(code2, "pe", "tradeDate=2018-04-25;ruleType=10",usedf=True)
  
  upstd = Winsorzationpestd['PE'].mean() + 3 * Winsorzationpestd['PE'].std()
  downstd = Winsorzationpestd['PE'].mean() - 3 * Winsorzationpestd['PE'].std()
  print(upstd,downstd )
  Winsorzationpestd['PE_std'] = Winsorzationpestd['PE'].apply(lambda x: upstd if x>upstd else (downstd if x<downstd else x))
  
  fig, ax = plt.subplots(figsize=(14, 5))
  Winsorzationpestd['PE'].plot(color="r")
  Winsorzationpestd['PE_std'].plot(color="b")
  plt.xlabel("stock code")
  plt.legend(["PE", "PE_std"])
  ax.grid()
  ```

  ![1536392265836](Photo\1536392265836.png)


* 分位数去极值

  将因子值进行升序的排序，对排位百分位高于 97.5%或排位百分位低于2.5% 的因子值，进行类似于  MAD  、 3σ  的方法进行调整  。

```python
def filter_extreme_percentile(series,min = 0.10,max = 0.90): #百分位法 
    series = series.sort_values() 
    q = series.quantile([min,max]) 
    return np.clip(series,q.iloc[0],q.iloc[1]) 

# 或者直接采用scipy的winsorize函数
error_code,data_Winsorzation3 = w.wset("sectorconstituent", "date=2018-04-25;sectorId=1000000090000000",usedf=True) 
#沪深300指数成分股
code = ','.join(data_Winsorzation3['wind_code'].values) #获取指数成分股代码
error_code,Winsorzationpe =w.wss(code, "pe", "tradeDate=2018-04-25;ruleType=10",usedf=True)

# 引入scipy包的winsorize方法
Winsorzationpe['PE_pct'] = stats.mstats.winsorize(Winsorzationpe['PE'], limits=0.025) # 置信区间95%

fig, ax = plt.subplots(figsize=(14, 5))
Winsorzationpe['PE'].plot(color="r")
Winsorzationpe['PE_pct'].plot(color="b")
plt.xlabel("stock code")
plt.legend(["PE", "PE_pct"])
ax.grid()
```

![1536393003891](Photo\1536393003891.png)



三种方法的主要区别就是对于异常值判断的度量值不同。顾名思义，MAD去极值法是以n个MAD为界，当元素与均值差的绝对值超过n个MAD时，该元素被认定为异常值，此时我们更改该元素为均值加上（或减去）n倍的MAD；Std法类似，唯一的区别是以标准差为标准判断是否为极值。分位数是稳健统计量，因此分位数方法对极值不敏感，但如果样本数据正偏严重，且右尾分布明显偏厚时，分位数去极值方法会把过多的数据划分为异常数据。

![1536393104925](Photo\1536393104925.png)



假定数据符合正态分布，且在95%置信区间内数据可靠，从上图可以看到，MAD保留了分位数缩尾的因子差异特性，同时进一步抑制了异常值的范围。但是具体情况也要具体讨论。



#### 4、标准化

大家在做量化时，很多时间花费在数据处理这块上，同时数据的质量也影响到了我们回测的精度和准度。

标准化（standardization）是将处理后的数据从有量纲转化为无量纲，从而使得数据更加集中，便于不同单位或量级的指标能够进行比较和加权。

主要的方法有普通标准化和行业标准化，区别是因子均值的计算方法不一样。

（**万矿主要介绍的是普通标准化，优矿的课程中有用行业均值做标准化的介绍**）



##### a、 min-max 标准化 (Min-max normalization) 

也叫离差标准化，是对原始数据的线性变换，使结果落到[0,1]区间，如下：

$$
\large x_{normalization}=\frac{x-min}{max-min}
$$
max：样本数据的最大值  
min：样本数据的最小值

实际使用中可以用经验常量值来替代max和min。而且当有新数据加入时，可能导致max和min的变化，需要重新定义。

```python
def MaxMinNormalization(x):  
    n = (x - np.min(x)) / (np.max(x)- np.min(x));  
    return n; 


MaxMin = MaxMinNormalization(alldata['PE'])

fig = plt.figure(figsize=(14,5))
MaxMin.plot()
plt.grid()
```

![1536394360383](Photo\1536394360383.png)



##### b、Z-score 0均值标准化(zero-mean normalization)

数据符合标准正态分布，即均值为0，标准差为1，如下：

$$
\large {x}_{Zscore}=\frac{(x-\mu )}{\sigma }
$$
u: 所有样本数据的均值  

σ: 为所有样本数据的标准差  

```python
def Z_ScoreNormalization(x):  
    n = (x - np.average(x)) / np.std(x);  
    return n

Z_Scor =  Z_ScoreNormalization(alldata['PE'])

fig = plt.figure(figsize=(14,5))
Z_Scor.plot()
plt.grid()
```

![1536394657109](Photo\1536394657109.png)



#### 5、缺失值处理

滤除缺失值

	DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

+ **`axis`**

    释义：0 or index，1 or columns，默认0


+ **`how`**

    释义： any，all，默认any。any：有一个NAN就算缺失；all：行或列全缺失才算缺失。


+ **`thresh`**

    释义： Require that many non-NA values.


+ **`subset`**

    释义： 标签名。选择对某个列或行进行检查缺失。即限制检查范围。


+ **`inplace`**

    释义： 修改调用对象而不产生副本。默认为False



如果不想滤掉缺失数据，而是希望通过其他方法填补，在Python中。我们可以用 `fillna` 方法。现在介绍如下：

	DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)  

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html?highlight=fillna#pandas.DataFrame.fillna  

```python
# 通过一个字典调用时，可以实现对不同的列填充不同的值
WKA_df.fillna(value={'OPEN':29,'HIGH':28})
```



#### 6、中性化

	未提及







###  单因子分析


拿到了数据，处理完之后我们就该进行因子分析，对于因子分析我们主要从以下几方面进行分析：

* IC分析：计算每一期因子值与下一期股票收益率之间的相关性（信息系数IC），并对IC序列进行进一步分析
* 收益率分析：每一期对股票池内的股票按因子值由优到劣进行排序，并分为N组，对每一组的收益率进行分析
* 换手率分析：通过每一期买入股票的变动，评价因子的交易成本暴露
* 版块分析：对于所选出来的股票的行业分布进行分析





#### 1、IC序列分析、IC衰减

IC分析在整个单因子分析的过程中起主导性作用
* IC的大小反应了因子暴露值与股票表现之间的线性关系，IC的值越大说明因子的预测性越高，选股的效果就越好。IC的计算方法主要有两种
  ，但常用的更有解释性的为RankIC

* IC_IR反应的是IC序列的稳定性，实际上类似于收益率的夏普比率，我们用IC时间序列的均值除以其方差得到IC_IR。所以ICIR的值越大越好

#####  ic_analysis()

万矿的因子分析函数


|     参数     | 类型         | 说明                                                         |
| :----------: | ------------ | ------------------------------------------------------------ |
| ind_ret_data | pd.DataFrame | 处理后的因子数据，结构如prepare_raw_data返回的数据           |
|  ic_method   | str          | ic计算方法，'rank':依据排序大小计算信息系数，'normal'为依据数值大小计算的信息系数 |

| 返回       |                                                              |
| ---------- | ------------------------------------------------------------ |
| ICAnalysis | 其中ic_series为每个因子ic时间序列，ic_decay为每个因子12期的IC衰减，ic_stats为ic的统计指标 |

```python
# 因子分析
ic_ana = wa.ic_analysis(processed_inds_ret)
ic_ana.ic_stats   # 因子的ic统计
ic_ana.ic_series  #因子ic时间序列
```

![1536396254929](Photo\1536396254929.png)

```python
# 这里要注意万得的API函数，wline,实际上跟plot的效果差不多，但是交互性更强，用户可以实时查看图像上的数据
ind = "PE_TTM"
fig_ic=WLine("IC序列：{}".format(ind),"2016.01-2018.03", ic_ana.ic_series.ix[ind])
fig_ic.plot()
```

![1536396331625](Photo\1536396331625.png)

#####  IC信号衰减

一般提到IC我们都是计算的当期的因子值与下一期收益率之间的相关性，也就是因子值与收益率之间相差一个周期。而IC衰减描述的是因子值和相隔LEG期的收益率的相关性。具体的计算方法如下：

如果一共有N期的因子数据和收益率数据，我们先把所有i期因子和i+1期收益率的IC值算出来求平均，再把i期因子和i+2收益率的IC求平均....（i=1,...,N-LAG),最终我们得到LAG个IC的均值，这几个均值就体现了IC的衰减。

```python
ic_ana.ic_decay
fig_decay=WBar('{} IC Decay'.format(ind), '',ic_ana.ic_decay[ind].to_frame())
fig_decay.plot()
```

![1536396699767](Photo\1536396699767.png)

####  2、收益率分析

**根据选股结果进行收益率分析（不使用回测框架）**

收益率分析也是对因子的进一步分析，在因子通过IC分析后，我们知道了因子对于股票收益的相对关系有预测性的作用，但具体反应到收益上还得根据收益率分析的相关指标来判断。

这里我们在做收益率分析时，我们采用分组的办法来具体的考察因子对于股票的区分度。如果一个因子的区分度够高，正如在第一部分时讲的，因子值排名靠前的那组的平均收益肯定要高于最后一组。我们在接下来的分析中，把股票分为了5组。实际操作中，我们也可以只选取排名靠前的百分之几的股票和排名靠后的百分几的股票进行比较分析。

由于运算效率的取舍问题，在因子筛选的初期，我们可以直接用每期股票的平均收益率来计算分析期间的分组累计收益率等指标，而不用写到策略中每日回测。





##### 分组函数：add_group

|     参数     |     类型     | 说明                                                         |
| :----------: | :----------: | ------------------------------------------------------------ |
| ind_ret_data | pd.DataFrame | 处理后的因子数据，结构如prepare_raw_data返回的数据           |
|   ind_name   |     str      | 需要分子的因子名                                             |
|  group_num   | int or float | 当为大于等于2的整数时，对股票平均分组；当为（0,0.5）之间的浮点数，对股票分为3组，前group_num%为G01，后group_num%为G02，中间为G03 |
|  direction   |     str      | 设置所有因子的排序方向，'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高； |
| industry_neu |     bool     | 分组时是否采用行业中性，默认False不采用行业中性              |

```python
# 使用分组函数，对得到的因子数据进行分组
group_data_ind = wa.add_group(processed_inds_ret,ind_name=ind, industry_neu=True)
group_data_ind
```

![1536397470614](Photo\1536397470614.png)

```python
group_data_ind.groupby(level=0).apply(lambda x: x.reset_index(level=0,drop=True).sort_values('GROUP'))
# 因为这里的dataframe有两级索引(日期和股票名称)，所以需要指定level，drop=True表示去掉索引列
```

![1536398054868](Photo\1536398054868.png)







##### 收益率分析函数： return_analysis

|     参数      |     类型     | 说明                                                         |
| :-----------: | :----------: | :----------------------------------------------------------- |
| ind_ret_data  | pd.DataFrame | 处理后的因子数据，结构如prepare_raw_data返回的数据           |
|  bench_code   |     str      | 基准代码，如'000300.SH'                                      |
|  start_date   |     str      | 数据开始日期，如'2015-01-01'                                 |
|   end_date    |     str      | 数据结束日期，如'2017-12-31'                                 |
|  ret_method   |     str      | 组合收益率加权方法， 'cap': 市值加权， 'equal':等权，默认'cap' |
|   group_num   | int or float | 当为大于等于2的整数时，对股票平均分组；当为（0,0.5）之间的浮点数，对股票分为3组，前group_num%为G01，后group_num%为G02,中间为G03 |
| ind_direction |     str      | 设置所有因子的排序方向，'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高；当为dict时，可以分别对不同因子的排序方向进行设置 |

| 返回           |                                                              |
| -------------- | ------------------------------------------------------------ |
| ReturnAnalysis | WindAlpha自带收益分析数据类型，不同属性的说明如下：group_mean_return：每组每期平均收益率return_stats：收益率的统计指标，group_cum_return：每组每期的累计收，benchmark_return：基准收益率 |

```python
# 使用return_analysis函数，计算分组后每组收益的平均值

# 按照函数说明，构造输入参数
direction_dict = {'PE_TTM': 'descending'}

ret_ana = wa.return_analysis(processed_inds_ret,'000300.SH','2018-01-01', '2018-04-30',ind_direction=direction_dict)

# 计算相关的收益率指标
ret_ana.return_stats.applymap(lambda x: round(x,4) if not isinstance(x, tuple) else (round(x[0],4), round(x[1],4)))
# 注意apply和applymap的区别
# applymap是一种让函数作用于DataFrame每一个元素的操作；
# apply是一种让函数作用于列或者行操作；
# round(x,4)保留四位有效数字
# isinstance(x, tuple) 查看x数据类型是否为元组
```

![1536397857609](Photo\1536397857609.png)







#### 3、换手率分析

即对于交易成本的考量

对于换手率的分析，常用的方法有两种：个数法和权重法。

所谓个数法就是计算每期之间股票变动的数量并除以股票的总数量计算出的比率，例如t期买入[A,B,C,D,E]五只股票，t+1期买入[A,D,E,F,G]五只股票，那么这期间的换手率就是(2/5=40%)。

而权重法不仅考虑股票本身的变化，还考虑了股票权重的变化。



#####   换手率分析函数：turnover_analysis()

| 参数          | 类型         | 说明                                                         |
| ------------- | ------------ | ------------------------------------------------------------ |
| ind_ret_data  | pd.DataFrame | 处理后的因子数据，结构如prepare_raw_data返回的数据           |
| method        | str          | 换手率计算方法，'count'个数法，'cap',权重法，默认'count'     |
| group_num     | int or float | 当为大于等于2的整数时，对股票平均分组；当为（0,0.5）之间的浮点数，对股票分为3组，前group_num%为G01，后group_num%为G02,中间为G03 |
| ind_direction | str          | 设置所有因子的排序方向，'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高；当为dict时，可以分别对不同因子的排序方向进行设置 |

| 返回             |                                                              |
| ---------------- | ------------------------------------------------------------ |
| TurnoverAnalysis | WindAlpha自带换手率分析数据类型，不同属性的说明如下：turnover：每组换手率数据时间序列，buy_signal：买入信号衰减与反转， auto_corr:因子自相关性 |



注意概念：（买入信号衰减与反转）

在因子分析过程中，我们经常看到衰减这个词，实际上很好理解，衰减的越快持续性越差，衰减的越慢持续性越好。买入信号的衰减可以和IC衰减配合着使用，其度量的是当前买入的股票在后续调仓期中买入的比率。如果一个因子的IC衰减和买入信号衰减都很慢，那么说明该因子的区分度很高，同时使用该因子选股的换手成本很低。

买入信号反转与衰减的原理类似，唯一的区别是统计的组别不一样。当对因子进行排序后，我们默认买入的是G1组（第一组），卖出G5组（最后一组），买入信号衰减度量的是当前买入的股票在后续调仓期卖出的比率。

```python
# 使用换手率函数对因子进行分析
turnover_ana=wa.turnover_analysis(processed_inds_ret)

# 画图表示因子的换手率时间序列
fig3=WLine("换手率：PE_TTM","2016.01-2017.12", turnover_ana.turnover.ix["PE_TTM"])
fig3.plot()
```

![1536399343449](Photo\1536399343449.png)





#### 4、板块分析

##### 选股结果分析函数：sector_analysis()

| 参数           | 类型         | 说明                                                         |
| -------------- | ------------ | ------------------------------------------------------------ |
| ind_ret_data   | pd.DataFrame | 处理后的因子数据，结构如prepare_raw_data返回的数据           |
| method         | str          | 换手率计算方法，'count'个数法，'cap',权重法，默认'count'     |
| group_num      | int or float | 当为大于等于2的整数时，对股票平均分组；当为（0,0.5）之间的浮点数，对股票分为3组，前group_num%为G01，后group_num%为G02,中间为G03 |
| ind_direction  | str          | 设置所有因子的排序方向，'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高；当为dict时，可以分别对不同因子的排序方向进行设置 |
| industry_type  | str          | sw表示申万行业分类，citic表示中信行业分类，默认'sw'          |
| industry_level | int          | 行业分类等级，默认1                                          |


| 返回           |                                                              |
| -------------- | ------------------------------------------------------------ |
| SectorAnalysis | WindAlpha自带选股结果分析数据类型，不同属性的说明如下：group_cap_mean：每组选出股票的市值均值,group_industry_mean_ratio：每组所有时间的行业平均占比,group_industry_ratio: 每组所行业占比时间序列,group_stock_list：每组选出股票的代码 |

```python
# 利用板块分析函数进行分析
code_ana = wa.sector_analysis(raw_inds_ret)

# group_industry_ratio,返回每组所行业占比时间序列
code_ana.group_industry_ratio.ix['PE_TTM'].ix['G05'].fillna(0)

# 可以通过wstacking_bar 画出交互图形
data =code_ana.group_industry_ratio.ix['PE_TTM'].ix['G01'].fillna(0)
chart = WStacking_bar(title='FA_NPGR_TTM因子选股行业占比', data=data.T, data_label=False)
chart.plot()
```

![1536491145267](Photo\1536491145267.png)

```python
# 获得某一因子所有时间段内的行业板块分组情况，然后通过饼状图的形式画出来。

ind_mean_G01=code_ana.group_industry_mean_ratio.ix['PE_TTM']['G01']
ind_mean_G01 = pd.DataFrame({'name':ind_mean_G01.index, 'value': ind_mean_G01.values})
fig5 = WPie('G01 Section Percent', '', ind_mean_G01)
fig5.plot()


ind_mean_G01=code_ana.group_industry_mean_ratio.ix['PE_TTM']['G01']
ind_mean_G01 = pd.DataFrame({'name':ind_mean_G01.index, 'value': ind_mean_G01.values})
fig5 = WPie('G01 Section Percent', '', ind_mean_G01)
fig5.plot()
```

![1536491205722](Photo\1536491205722.png)



### 多因子组合分析

#### 多因子组合打分法

	**多因子组合分析与单因子分析主要多出了以下两个过程：**

* 因子选择的过程：静态选择和动态选择

* 单因子得分到多因子组合得分的过程，这个过程涉及到了各单因子得分该如何加总的问题

主要的组合得分计算有以下几种方法：

* 等权法：该方法对所有因子同等看待，不论其有效性的优劣
* IC加权：根据IC均值的大小决定因子的权重，IC高的因子，权重就大，IC的均值为滚动计算
* ICIR加权：根据因子ICIR的大小决定因子的权重，ICIR越大，权重越大，ICIR的值为滚动计算



##### 打分法函数 ：score_indicators()

| 参数          | 类型         | 说明                                                         |
| ------------- | ------------ | ------------------------------------------------------------ |
| ind_ret_data  | pd.DataFrame | 处理后的因子数据，结构如prepare_raw_data返回的数据           |
| score_method  | str          | 打分方法，可选有'equal':因子等权，'ic':因子ic加权，'icir':因子icir加权 |
| ind_direction | str          | 设置所有因子的排序方向，'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高；当为dict时，可以分别对不同因子的排序方向进行设置 |
| ic_window     | int          | ic或icir打分法时ic计算均值及标准差的数据量                   |

```python
import pandas as pd
import WindAlpha as wa

raw_inds_ret = pd.read_csv('data/ind_data.csv', index_col=[0,1])
raw_inds_ret.index.names = ['date','codes']
processed_inds_ret = wa.process_raw_data(raw_inds_ret)
df_score = wa.score_indicators(processed_inds_ret)

df_score
```

![1536500524168](Photo\1536500524168.png)

```python
# 将打好分的股票进行选股分组处理
score_code_ana = wa.sector_analysis(df_score)
code_list=score_code_ana.group_stock_list.ix['SCORE']#索引行数据
```

![1536502222732](Photo\1536502222732.png)

```python
stock_G01 = code_list['G01']
stock_G05 = code_list['G05']

from WindPy import *
from datetime import *
import pandas as pd
w.start(show_welcome=False)

list_A = w.wset("SectorConstituent",u"date=20130608;sector=全部A股").Data[1]      # 全部A股  作为初始股票池

from WindAlgo import * #引入回测框架

def initialize(context):             #定义初始化函数
    global stock_G05
    context.capital = 1000000        #回测的初始资金
    context.securities = list_A      #回测标的 这里是全部A股
    context.start_date = "20170101"  #回测开始时间
    context.end_date = "20180330"    #回测结束时间
    context.period = 'd'             # 'd' 代表日, 'm'代表分钟   表示行情数据的频率
    context.benchmark = '000300.SH'  #设置回测基准为沪深300
    context.stock_list = stock_G05   # 后20%
    

def handle_data(bar_datetime, context, bar_data):
    pass
    
def my_schedule1(bar_datetime, context, bar_data):             

    bar_datetime_str = bar_datetime.strftime('%Y-%m-%d')           #设置时间

    code_list = list(context.stock_list[bar_datetime_str])  #选择最后20%的股票 即因子最大的20%的股票
    wa.change_securities(code_list)
    context.securities = code_list    #改变证券池 
    
    list_sell = list(wa.query_position().get_field('code'))     
    #此处可改进 有些下个月打算买的股票可以不用卖出只需要调仓即可 可节省一些手续费
    for code in list_sell:
        volumn = wa.query_position()[code]['volume']    #找到每个code 的 持仓量 
        res = wa.order(code,volumn,'sell',price='close', volume_check=False)  # 卖出上一个月初 买入的所有的股票
    ## '卖出上个月所有仓位'  为本月的建仓做准备  

def my_schedule2(bar_datetime, context,bar_data):
    # 在单因子选股的结果中 剔除 没有行情的股票   ？？？
    buy_code_list=list(set(context.securities)-(set(context.securities)-set(list(bar_data.get_field('code')))))  
 
    for code in buy_code_list:
        res = wa.order_percent(code,1/len(buy_code_list),'buy',price='close', volume_check=False)  
        #对最终选择出来的股票建仓 每个股票仓位相同   '本月建仓完毕'

wa = BackTest(init_func = initialize, handle_data_func=handle_data)   #实例化回测对象
wa.schedule(my_schedule1, "m", -1)         #   m表示在每个月执行一次策略 0表示偏移  表示月初第一个交易日往后0天
wa.schedule(my_schedule2, "m", 0)         #在月初第2个交易进行交易
res = wa.run(show_progress=True)          #调用run()函数开始回测,show_progress可用于指定是否显示回测净值曲线图
nav_df = wa.summary('nav')                  #获取回测结果  回测周期内每一天的组合净值
```

