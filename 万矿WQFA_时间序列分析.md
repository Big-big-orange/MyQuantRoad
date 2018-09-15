

## 时间序列分析

课程链接：https://www.windquant.com/qntcloud/college

### 

- 本小节自己学到的小技巧

   

  自相关性常用来解释金融系统中经济行为在时间上的惯性，比如人们消费的行为会受到习惯的影响

  偏自相关是为了衡量过去单期对现在的影响，剔除其他期的作用而引入的。



  QQ图可以：

  1、两组数据是否来自同一分布，将其中一组数据作为参考，另一组数据作为样本。

  2、检验一组数据是否服从某一分布



  峰度和偏度

  ​	峰度是三阶矩，偏度是4阶矩



### 时间序列简介

金融数据分析中，我们最常见的有数据类有三种，分别是：**`横截面数据（Cross Sectional Data）`**、**`时间序列数据（Time Series Data）`**、**`面板数据（Panel Data）`**。

下面，我们结合金融数据分析理论和量化投资研究对三者做一个简单的介绍：

#### 横截面数据（Cross Sectional Data）

1、指在同一时间（时期或时点）截面上反映一个总体的一批（或全部）个体的同一特征变量的观测值，也称静态数据。它对应同一时点上不同空间(对象)所组成的一维数据集合，研究的是某一时点上的某种经济现象，突出空间(对象)的差异。比如可以从不同公司在同一时间发布的财务报表中，得到同一年度这些公司的一些财务数据。

2、比如，我们通过 *w.wss* 获取选定证券品种的历史截面数据，比如取HS300股票中部分股票在2017年3季度的净利润财务指标数据。

```python
from WindPy import *
w.start()

# 浦发银行、平安银行
error_code,data_df = w.wss("600000.SH,000001.SZ","eps_ttm,surpluscapitalps","rptDate=20171231",usedf=True)
data_df
```

![1536503802831](Photo\1536503802831.png)



#### 时间序列数据（Time Series Data）

1、时间序列数据是指对同一对象在不同时间连续观察所取得的数据。它着眼于研究对象在时间顺序上的变化，这类数据反映了某一事物、现象等随时间的变化状态或程度。

2、与横截面数据相比较，其区别在与组成数据列的各个数据的排列标准不同。时序数据是按时间顺序排列的，横截面数据是按照统计单位排列的。因此，横截面数据不要求统计对象及其范围相同，但要求统计的时间相同。也就是说必须是同一时间截面上的数据。

3、我们可以通过 *w.wsd* 获取选定证券品种某一段时间内的历史序列数据。包括日间的行情数据、基本面数据以及技术数据指标。

4、例如获取 000001.SZ 从 2017-01-01 至 2018-05-01 间所有高开低收等指标的数据。

```python
error_code,data1_df = w.wsd("000001.SZ","open,high,close,low", "2017-01-01", "2018-05-01", "",usedf=True)
data1_df.head()
```

![1536503918071](Photo\1536503918071.png)

#### 面板数据（Panel Data）

也称平行数据，是截面数据与时间序列数据综合起来的一种数据类型。指在时间序列上取多个截面，在这些截面上同时选取样本观测值所构成的样本数据。或者说他是一个m*n的数据矩阵，记载的是n个时间节点上，m个对象的某一数据指标。简而言之就是对上面两个数据的综合。





### 金融时间序列的线性模型

#### 相关系数

线性相关的程度常用皮尔逊（Pearson）相关系数来衡量。在统计上，两个随机变量 $X$ 和 $Y$ 的相关系数定义为为： 
$$
\large \rho_{xy} = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}=\frac{E[(X- \mu _{x})(Y-\mu _{y})]}{\sqrt{ E((X- \mu _{x})^{2}E(Y-\mu _{y})^{2}}}
$$
而我们的根据样本的估计计算公式为

$$
\large \rho_{xy} = \frac{\sum_{t=1}^{T}(x_t- \bar x)(y_t-\bar y)}{\sqrt{ \sum_{t=1}^{T}(x_t- \bar x)^{2}\sum_{t=1}^{T}(y_t- \bar y)^{2}}}
$$
其中 $\bar{x}$ 和 $\bar{y}$ 分别是 $X$ 和 $Y$ 的样本均值。并且假定方差是存在的。这个系数是度量 $X$ 和 $Y$ 线性相关的程度。

完全线性正相关意味着相关系数为 +1.0，完全线性负相关意味着相关系数为 -1.0，其他情况下相关系数在 -1.0 和 +1.0 之间。绝对值越大表明相关性越强。

我们也可以通过scipy库中的函数，计算上面的Pearson系数的大小：

```python
from scipy import stats
stats.pearsonr(pct_hs300_log[0:192], pct_gzmt_log)[0]
```



#### 自相关系数(ACF)

相关系数衡量了两个序列的线性相关程度，而自相关函数，顾名思义就是衡量自己和自己的相关程度，即 r(t) 和过去某个时间 t(t−l) 的相关性：

考虑平稳时间序列 $r_t$，$r_t$ 与 $r_{t-l}$ 的相关系数称为$r_t$ 的**`间隔为 l 的自相关系数，通常记为 ρl。具体的定义：`**

$$
\large \rho_l = \frac{Cov(r_t,r_{t-l})} {\sqrt{Var(r_t)Var(r_{t-l})}} = \frac{Cov(r_t,r_{t-l})} {Var(r_t)}=\frac{\gamma_{l}}{\gamma_{0}}
$$
**这里用到了弱平稳序列的性质:**
$$
\large Var(r_t)=Var(r_{t-l})
$$
根据定义，$ρ_0=1$，$ρ_l=ρ_l$，和 -1 ≤ ρl ≤ 1。自相关系数组成的集合 ${ρ_l}$ 称为 $r_t$ 的自相关函数（Autocorrelation Function）。一个若平稳的时间序列是序列自身前后不自相关的。

对一个**`平稳时间序列`**的样本 ${r_t}$， 1 ≤ t ≤ T，则间隔为 l 的样本自相关系数的估计为：

$$
\large \hat \rho_l = \frac{\sum_{t=l+1}^{T}(r_t- \bar r)(r_{t-l}-\bar r)}{ \sum_{t=1}^{T}(r_t- \bar r)^{2}}, 0 \leqslant l \leqslant T-1 
$$
序列的自相关性常用来解释金融系统中经济行为在时间上的惯性，比如人们消费的行为会受到习惯的影响，并不会由于收入的增加或减少而立刻调整。呈现出一定程度的自我相关。

在Python中，我们可以使用statsmodels包中的 *`acf()`* 函数来计算时间序列的自相关系数。

```PYTHON
statsmodels.tsa.stattools.acf(x, unbiased=False, nlags=40, qstat=False, fft=False, alpha=None, missing='none')
```

**输入**


+ **`x`**      :                    释义：时间序列数据。


+ **`unbiased`**        释义：如果为True，会对分母进行调整为 n-k，使得结果为无偏的估计值。默认为 n。


+ **`nlags`**                释义：自相关系数的最大滞后期数


+ **`qstat`**                 释义：acf()函数是否返回 Ljung-Box 检验结果。


+ **`ff`**                        释义：表明使用的算法，如果为True，计算ACF使用FFT（快速傅氏变换）。默认为False。


+ **`alpha`**                 释义：置信区间所用的置信水平。默认为None，即不计算置信区间。例如：alpha＝0.05, 返回95%的置信区间。


+ **`missing`**            释义：NANs如何处理，有 [‘none’, ‘raise’, ‘conservative’, ‘drop’] 供去选择。

**输出**


+ **`acf `**                       释义：自相关函数。


+ **`confint `**             释义：置信区间


+ **`qstat `**                  释义：Ljung-Box Q-Statistic 检验结果，如果 qstat 是 True。


+ **`pvalues `**            释义：与Q统计量相关的 p-values 值，如果 qstat 是 True。



举例：检验上证综指收盘价的自相关系数

```python
error_code,close_szzz = w.wsd("000001.SH", "close,pct_chg", "2016-01-01", "2017-01-01", "",usedf=True)

# 取上证综指的收盘价，计算自相关函数，检验统计量Q以及p值。

from scipy import  stats
import statsmodels.api as sm 
from matplotlib import pyplot
from statsmodels.graphics.api import qqplot
# 引入qq图，比较两组数据是否服从同一分布
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data_close = close_szzz['CLOSE'] 
m = 30 # 检验30个自相关系数
acf,q,p = sm.tsa.stattools.acf(data_close, nlags=m, qstat=True)  
x = np.c_[range(1,31), acf[1:], q, p] # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
xput=pd.DataFrame(x, columns=['lag', "ACF", "Q-Statistic", "p-values"])
xput = xput.set_index('lag')
xput.head().append(xput.tail())
```

![1536582007535](Photo\1536582007535.png)

```python
fig, ax = plt.subplots(figsize=(14,5))
fig = plot_acf(data_close[0:30], alpha=0.05, ax=ax)
```

![1536583316694](Photo\1536583316694.png)





#### 偏相关系数(PACF)

假设股票价格偏p1，p2，···，pt的一阶自相关系数 ρ1 大于0，即今天的价格 pt 与昨天的价格 pt-1 相关，可能 pt 也会受前天，大前天的价格的影响。也就是说 pt 与 pt-1 的 ACF 算出的结果不单纯是昨天对今天的结果，而是包含了之前的一些信息，间接地对今天产生了影响。为了衡量过去单期对现在的影响，剔除其他期的作用，引入 PACF。

具体比较复杂，后面介绍AR模型的时候我们再详细论述。

在Python中，我们可以使用statsmodels包中的 *pacf()* 函数来计算时间序列的偏自相关系数。

```PYTHON
statsmodels.tsa.stattools.pacf(x, nlags=40, method='ywunbiased', alpha=None)
```

**输入**


+ **`x`**                   释义：时间序列数据。


+ **`nlags`**        释义：偏自相关系数的最大滞后期数


+ **`method`**      释义：指定计算使用的方法（'ywunbiased', 'ywmle', 'ols）：

    yw or ywunbiased : yule walker with bias correction in denominator for acovf. Default.

    ywm or ywmle : yule walker without bias correction

    ols - regression of time series on lags of it and on constant

    ld or ldunbiased : Levinson-Durbin recursion with bias correction

    Ldb or ldbiased : Levinson-Durbin recursion without bias correction


+ **`alpha`**         释义：置信区间所用的置信水平。默认为None，即不计算置信区间。例如：alpha＝0.05, 返回95%的置信区间。

**输出**


+ **`pacf`**                 释义：自相关函数。


+ **`confint`**          释义：PACF的置信区间

```PYTHON
m1 = 30 # 检验30个偏自相关系数
pacf = sm.tsa.stattools.pacf(data_close,nlags=m1, method='ywunbiased',alpha=None)  
x1 = np.c_[range(1,31), pacf[1:]]
xput1 = pd.DataFrame(x1, columns=['lag', "PACF"])
xput1 = xput1.set_index('lag')
xput1.head().append(xput1.tail())
```

![1536583786631](Photo\1536583786631.png)

```python
# 取上证综指的收盘价，计算偏自相关函数
fig, ax = plt.subplots(figsize=(14,5))
fig = plot_pacf(data_close[0:30], ax=ax)
```

![1536584158310](Photo\1536584158310.png)



偏自相关性图反映了时间序列的各自偏自相关系数的大小。图中黑色柱子的高度对应的是各阶偏自相关系数的值。上图图，基本上偏自相关系数都落在了95%的置信区间。只有2阶偏自相关系数落在95%置信区间外。即，2阶滞后收盘价与当期收盘价的相关性是显著的。





### 平稳性

在时间序列分析中，平稳性是时间序列分析的基础。

时间序列的平稳性是其基本的假设，只有基于平稳时间序列的预测才是有效的。平稳性有强平稳和弱平稳之分，在一般条件下，我们所说的平稳时间序列指的是弱平稳时间序列。

我们先看第一个例子，来直观的感受一下平稳性。

下图中第一张图为上证综指部分年份的收盘价，是一个非平稳时间序列；第二张图是其收益率。而第三第四张图为收益率的n阶差分，为平稳时间序列。

```python
close_szzz['closeDiff_1'] = close_szzz['CLOSE'].diff(1)  # 1阶差分
close_szzz['closeDiff_2'] = close_szzz['closeDiff_1'].diff(1)  # 2阶差分
fig, ax = plt.subplots(figsize=(14,9))
close_szzz.plot(subplots=True,ax=ax)
```

![1536584588242](Photo\1536584588242.png)

平稳性是金融时间序列分析的基础。同时，平稳性又分为了严平稳（strictly stationary）和弱平稳（weekly stationary）。

#### 严平稳（strictly stationary）

如果对所有的 t，任意正整数 k 和任意 k 个正整数

$$
\large (t_1,t_2......t_k), \large (r_{t_1},r_{t_2},......r_{t_k})
$$
的联合分布与

$$
\large (r_{t_1 + t},r_{t_2+t},......r_{t_k + t}) 
$$

的联合分布相同，我们称时间序列 {rt} 是**`严平稳的（strictly stationary）`**

换言之，严平稳性要求

\large (r_{t_1},r_{t_2},......r_{t_k})

的联合分布在时间的平移变换下保持不变。这是一个很强的条件，难以用经验方法验证。经常假定的是平稳性的一个较弱的方式。下面我们重点介绍一下弱平稳（weekly stationary）。



####  弱平稳（weekly stationary）

举例，可以看上证综指的对数收益率时间序列

![1536585717925](Photo\1536585717925.png)

我们看到上证综指在1996年1月1日至2018年1年1间的月对数收益率在0值上下变化。在统计上，**`这种现象表明收益率的均值不随时间变化，或者说，期望收益率具有时间不变性。`**上图也印证了这一点，除了在1990年至1995年的波动外，月对数收益率的范围大约在区间 [-0.2,0.2]。在统计上，**`该特征表明对数收益率的方差不随时间变化。`**

把上述两者结合在一起，我们就称对数收益率序为弱平稳的（weekly stationary）。

正式的说：

如果一个时间序列 ${X_t}$ 的**`一阶矩和二阶矩（均值和方差）具有时间不变性`**，则称它为弱平稳的（weekly stationary）。

弱平稳为预测提供了基础的框架。我们有理由相信沪深300未来的月收益率大约在0值左右，并且在 [-0.2,0.2]之间变化。

**`在金融文献中，通常假定资产收益率序列是弱平稳的，只要有足够多的历史数据，这个假定可以用实证方法验证。`**





#### 如何检验平稳性

##### 1、观察时间序列图，观察ACF和PACF图

对于平稳序列来说，ACF和PACF都会快速减小至0附近。或在某一阶后变为0.非平稳时间序列则是慢慢下降，不是快速减小。

![1536586237881](Photo\1536586237881.png)

![1536586271697](Photo\1536586271697.png)

##### 2、单位根检验

上面我们通过ACF、PACF图来判断时间序列的平稳性时，是以最直观的角度去看的，多多少少会有一些差异。为了更加客观、准确的检验时间序列的平稳性，为大家介绍一种统计检验，即单位根检验。

常见的单位根检验方法有`DF检验（Dickey-Fuller Test）`，`ADF检验（Augmented Dickey-Fuller Test）`和`PP检验（Phillips-Perron Test）`。本篇我们主要介绍ADF检验（Augmented Dickey-Fuller Test），其他的两个检验我们在后续会逐一讲解。

ADF的原假设为序列有单位根（非平稳）H0，备择假设为序列是平稳的H1。对于一个平稳的时序数据，就需要在给定的置信水平上显著，拒绝原假设。

Python中可使用statsmodels库中的adfuller来实现ADF检验。

```python
statsmodels.tsa.stattools.adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
```

**输入**

- `x`                                   释义：一维数组。

- `maxlag`                     释义：测试中包含的最大滞后，默认12 *（NOBS / 100）^ { 1/4 }

- `regression`           释义：回归中包含的常数和趋势阶（'c','ct','ctt','nc'）：
  - c:只有常量
  - ct:有常量项和趋势项
  - ctt:有常量项、线性和二次趋势项
  - nc:无任何选项

- `autolag`                 释义：（'AIC', 'BIC', 't-stat', None）
  - if None, then maxlag lags are used
  - if ‘AIC’ (default) or ‘BIC’, then the number of lags is chosen to minimize the corresponding information criterion
  - t-stat’ based choice of maxlag. Starts with maxlag and drops a lag until the t-statistic on the last lag length is significant using a 5%-sized test

- `store`                       释义：如果为True，则将结果实例附加返回到ADF统计。默认为False

- `regresults`         释义：如果为如果为True，则返回完整的回归结果。默认为False

**输出**

- `adf`                           释义：Test Statistic

- `pvalue`                   释义： p-value

- `usedlag`                释义：lags数量

- `nobs`                        释义：ADF回归的观测值和临界值

- `critical values`         释义：在1%、5%和10%水平的测试统计量的临界值。

- `icbest`                  释义： The maximized information criterion if autolag is not None.

- `resstore                释义： A dummy class with results attached as attributes



举例对上证综指的收盘价进行单位根检验

```python
import statsmodels.tsa.stattools as ts
def ADF(data):
    adftest = ts.adfuller(data)
    result = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key,value in adftest[4].items():
        result['Critical Value (%s)'%key] = value
    return result


ss = pd.DataFrame(ADF(data_close))
ss.columns=['data_close']
ss
```

![1536586931958](Photo\1536586931958.png)

Test Statistic Value 为-2.27，大于Critical Value(1%，5%，10%)显著性水平下的临界值，因此无法拒绝原假设，说明上证综指的收盘价序列式非平稳的。





### 金融数据的描述性统计分析

#### 概念

描述性统计，是指运用制表和分类，图形以及计算概括性数据来描述数据特征的各项活动。描述性统计分析要对调查总体所有变量的有关数据进行统计性描述，主要包括数据的频数分析、集中趋势分析、离散程度分析、分布以及一些基本的统计图形。

- 数据的频数分析。在数据的预处理部分，利用频数分析和交叉频数分析可以检验异常值。

- 数据的集中趋势分析。用来反映数据的一般水平，常用的指标有平均值、中位数和众数等。

- 数据的离散程度分析。主要是用来反映数据之间的差异程度，常用的指标有方差和标准差。

- 数据的分布。在统计分析中，通常要假设样本所属总体的分布属于正态分布，因此需要用偏度和峰度两个指标来检查样本数据是否符合正态分布。

- 绘制统计图。用图形的形式来表达数据，比用文字表达更清晰、更简明。包括条形图、饼图和折线图等。

通常我们通过有限样本数据来分析推测总体特征。





#### 均值

$$
\large\mu = \frac{\sum_{i=1}^N X_i}{N}
$$

$X_1, X_2, \ldots , X_N$ 是我们的观测值。

某些情况下样本的重要程度不一样，此时我们就需给每个样本赋予一定的权重比例，然后再计算其平均值，这种算法称为**样本加权平均数：**
$$
\large\sum_{i=1}^n w_i X_i 
$$

$$
\large\mu = \frac{\sum_{i=1}^N w_iX_i}{N}
$$

所有样本的权重和为$\sum_{i=1}^n w_i = 1$。 $w_i$为观测样本观测值$X_i$的权重。



##### 成交量加权平均价格（VWAP）

VWAP（Volume-Weighted Average Price，成交量加权平均价格）是一个非常重要的经济学量，它代表着金融资产的“平均”价格。某个价格的成交量越高，该价格所占的权重就越大。VWAP就是以成交量为权重计算出来的加权平均值，常用于算法交易。其属于被动型算法交易。

`VWAP策略是一种拆分大额委托单，在约定时间段内分批执行，以期使得最终买入或卖出成交均价尽量接近这段时间内整个市场成交均价`的交易策略。它是量化交易系统中常用的一个基准。作为一个基准量，VWAP就是一个计算公式：
$$
\large VWAP=\frac{\sum_{i=1}^{n}price_{i}*volume_{i}}{\sum_{i=1}^{n}volume_{i}}
$$
要做到这一点，VWAP模型必须把母单分割成为许多小的子单，并在一个指定的时间段内逐步送出去。这样做的效果就是降低了大单对市场的冲击，改善了执行效果；同时增加了大单的隐秘性。显然，VWAP模型的核心就是如何在市场千变万化的情况下，有的放矢地确定子单的大小、价格和发送时间。

VWAP模型对于在几个小时内执行大单的效果最好。在交易量大的市场中，VWAP效果比在流动性差的市场中要好。在市场出现重要事件的时候往往效果不那么好。如果订单非常大，譬如超过市场日交易量的1%的话，即便VWAP可以在相当大的程度上改善市场冲击，但市场冲击仍然会以积累的方式改变市场，最终使得模型的效果差于预期。

VWAP算法交易的目的是最小化冲击成本，并不寻求最小化所有成本。理论上，在没有额外的信息，也没有针对股票价格趋势的预测的情况下，VWAP 是最优的算法交易策略。

```python
(data_df['VOLUME'] * data_df['CLOSE']).sum() / data_df['VOLUME'].sum()
#或者
vwap = np.average(data_df['CLOSE'],weights=data_df['VOLUME']) #第一个参数为收盘价，并第二个参数为设定参考的权重为，也就是成交量 。numpy.average——沿着指定的轴计算加权平均值。
print(vwap)
```

##### 时间加权平均价格（TWAP）

TWAP（Time Weighted Average Price，时间加权平均价格）模型是把一个母单的数量平均地分配到一个交易时段上。该模型将交易时间进行均匀分割，并在每个分割节点上将拆分的订单进行提交。例如，可以将某个交易日的交易时间平均分为 N 段，TWAP 策略会将该交易日需要执行的订单均匀分配在这 N 个时间段上去执行，从而使得交易均价跟踪 TWAP，也是一个计算公式：
$$
\large TWAP=\frac{\sum_{i=1}^{n}price_{i}}{n}
$$
TWAP不考虑交易量的因素。TWAP的基准是交易时段的平均价格，它试图付出比此时段内平均买卖差价小的代价执行一个大订单。TWAP模型设计的目的是使交易对市场影响减小的同时提供一个较低的平均成交价格，从而达到减小交易成本的目的。在分时成交量无法准确估计的情况下，该模型可以较好地实现算法交易的基本目的。但是使用TWAP过程中的一个问题是，在订单规模很大的情况下，均匀分配到每个节点上的下单量仍然较大，当市场流动性不足时仍可能对市场造成一定的冲击。另一方面，真实市场的成交量总是在波动变化的，将所有的订单均匀分配到每个节点上显然是不够合理的。因此，算法交易研究人员很快建立了基于成交量变动预测的 VWAP 模型。不过，由于 TWAP 操作和理解起来非常简单，因此其对于流动性较好的市场和订单规模较小的交易仍然适用。

最简单的方法就是用arange函数创建一个从0开始依次增长的自然数序列，自然数的个数即为收盘价的个数。



#### 取值范围

即数据的极值以及完整的取值范围——最大值和最小值。

NumPy中有一个`ptp`函数可以计算数组的取值范围。该函数返回的是数组元素的最大值和最小值之间的差值。也就是说，返回值等于max(array) - min(array)。调用ptp函数：

```python
print ("最高价的最大值与最小值的差：", np.ptp(data_df['HIGH']))
print ("最低价的最大值与最小值的差：", np.ptp(data_df['LOW']))
```



#### 中位数（Median）

中位数（又称中值），代表一个样本、种群或概率分布中的一个数值，其可将数值集合划分为相等的上下两部分。

#### 众数（Mode）

众数是在一组数据中,出现次数最多的数据，是一组数据中的原数据，而不是相应的次数。如果有两个或两个以上个数出现次数都是最多的，那么这几个数都是这组数据的众数。

一组定量数据，统计每一个数值发生的次数，我们把它叫做**频数**，若将每一个数值发生的次数除以样本总量，得到了**频率**。将这些频数综合在一起，就得到了频数分布。



enumerate()是python的内置函数。enumerate在字典上是枚举、列举的意思。对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值。enumerate多用于在for循环中得到计数。

```python
hist, bins = np.histogram(data_df['CLOSE']) # bin:给定范围内的等宽度箱的数量，默认值为10
for i, j in enumerate(hist):
    print(i,j)
```





#### 几何平均值（Geometric mean）

$$
\large G = \sqrt[n]{X_1X_1\ldots X_n} 
$$

其中$X_i$均大于等于0。

我们也可以将它写成算术平均的对数形式：
$$
\large\ln G = \frac{\sum_{i=1}^n \ln X_i}{n} 
$$
几何平均值总是小于或等于算术平均值（所观测的数据为非负时），当所有数据都一样时等号成立。



有一种情况是，我们的收益率存在负数的时候该如何计算呢？

首先我们知道，收益率最低的值为 -100%，即 -1，那么我们可以对每一个收益率都 +1，这样就可以把负数消去。计算出新的几何平均值，然后再-1即可，如下：
$$
\large R_G = \sqrt[T]{(1 + R_1)\ldots (1 + R_T)} - 1
$$

```python
ratios =data_df['PCT_CHG']/100 + np.ones(len(data_df['PCT_CHG']))
R_G = stats.gmean(ratios) - 1
print('000001.SZ1 收益率的几何平均值:', R_G)
```



#### 调和平均数（Harmonic mean）

调和平均数又称倒数平均数，是总体各统计变量倒数的算术平均数的倒数。调和平均数是平均数的一种。调和平均数也有简单调和平均数和加权调和平均数两种。

调和平均数不如以上几种使用频繁，其公式为：
$$
\large H = \frac{n}{\sum_{i=1}^n \frac{1}{X_i}} 
$$
和几何平均数一样，我们可以重写调和平均值，使其看起来像一个算术平均值。调和平均值的倒数就是观测值的倒数的算术平均数：
$$
\large \frac{1}{H} = \frac{\sum_{i=1}^n \frac{1}{X_i}}{n} 
$$

#### 差分（Difference）

在学术文献中，收盘价的分析常常是基于股票收益率和对数收益率的。简单收益率是指相邻两个价格之间的变化率，而对数收益率是指所有价格取对数后两两之间的差值。因此，对数收益率也可以用来衡量价格的变化率。注意，由于**收益率**是一个比值，因此它是**无量纲**的。

要计算收益率我们可以直接在数据中提取，如果想进行验证计算，那么NumPy中的`diff`函数可以返回一个由相邻数组元素的差值构成的数组，即简单的收益率。这有点类似于微积分中的微分。为了计算收益率，我们还需要用差值除以前一天的价格。不过这里要注意，diff返回的数组比收盘价数组少一个元素。

```python
returns = np.diff(data_df['CLOSE']) / data_df['CLOSE'][ : -1] #没有用收盘价数组中的最后一个值做除数
(returns*100).head()

#对数收益率
logreturns = np.diff(np.log(data_df['CLOSE']))
pd.DataFrame(logreturns).head()
```

年波动率=（对数收益率标准差/均值）/（交易日倒数平方根)，通常交易日取252天。

```python
annual_volatility = np.std(logreturns)/np.mean(logreturns)
annual_volatility = annual_volatility / np.sqrt(1./252.)
print(annual_volatility)
```





#### 方差和标准差（Variance and Standard Deviation）

方差是在概率论和统计方差衡量随机变量或一组数据时离散程度的度量。概率论中方差用来度量随机变量和其数学期望（即均值）之间的偏离程度。统计中的方差（样本方差）是各个数据分别与其平均数之差的平方的和的平均数。在许多实际问题中，研究方差即偏离程度有着重要意义。

方差还可以告诉我们投资风险的大小。那些股价变动过于剧烈的股票会给持有者造成收益的波动。

方差的定义为：
$$
\large \sigma^2 = \frac{\sum_{i=1}^n (X_i - \mu)^2}{n} 
$$
标准差是方差的算数平方根。





#### 峰度和偏度

#####  一阶中心矩——均值

矩也称为动差。以**原点**为中心矩称为原点k阶矩，基本形式为：
$$
\large \mu =\frac{\sum_{i=1}^{n}x_{i}^{k}}{N}
$$


- 当 k=1 时，即1阶原点矩就是算术平均数。

- 当 k=2 时，即2阶原点矩就是平方平均数。



##### 二阶中心矩——方差

如果将原点移至算术平均数得位置，可以得到以平均数为中心的k阶中心矩：

$$\large m_{k} =\frac{\sum_{i=1}^{n}(x_{i}-\overline{x})^{k}}{N}$$

- 当 k=0 时，即零阶中心矩 m0=1

- 当 k=1 时，即一阶中心矩 m0=0，即均值。

- 当 k=2 时，即二阶中心矩 m0=σ2，即方差。
- 

##### 三阶中心矩——偏度（Skewness）

仅仅通过均值和方差来描述样本的分布函数往往是不够的，尽管正态分布完全由均值和方差就可以刻画，但是更多的分布函数往往做不到。所以高阶矩往往能够用来进一步刻画样本分布情况。

偏度(Skewness)是概率分布的三阶中心矩，来**衡量分布不对称或者倾斜的程度**。偏度(Skewness)亦称偏态、偏态系数。

与期望和标准差不同，**偏度没有单位**，是无量纲的量。取值通常在-3~+3之间，其绝对值越大，表明偏斜程度越大。

对于随机变量X，偏度(Skewness)定义式如下，其中k2，k3分别表示二阶和三阶中心矩：
$$
\large Skew(X)=E\begin{bmatrix} (\frac{X-\mu }{\sigma })^{3} \end{bmatrix}=\frac{k_{3}}{\sigma_{3} }=\frac{E[(X-\mu )^{3}]}{(E[(X-\mu )^{2}])^{3/2}}=\frac{k_{3}}{k_{2}^{3/2}}
$$
而对于样本的偏度，我们一般简记为SK，且分子分母都是无偏估计量，有：
$$
\large SK=\frac{k_{3}}{k_{2}^{3/2}}=\frac{n}{(n-1)(n-2)}\sum_{i=1}^{n}(\frac{x_{i}-{\mu}}{\sigma })^{3}
$$
其中 k2，k3 分别表示二阶和三阶累积无偏估计量，n 是观测个数，μ 是算数平均值，σ 是标准差。

正态分布的偏度为0，两侧尾部长度对称。同时偏度又分为正（右）偏、负（左）偏分布。

- Positive Skew：正（右）偏 > 0，表示一个尾部向正值方向延伸的不对称分布。其尾部比左侧更长或更胖。

- Negative Skew：负（左）偏 < 0，表示一个尾部向负值方向延伸的不对称分布。其尾部比右侧更长或更胖。

![1536590279911](Photo\1536590279911.png)

**在投资中，对冲基金的收益率通常被假定为正偏或负偏。其偏移方向，可以帮助我们评估一个给定的（或未来）收益率或价格是否高于或者低于其期望。**

**如果一个特定的收益率分布是偏的，那么它将有很大概率高于或低于符合正态分布的收益率。**

例如：
$$
\large Skew=E\begin{bmatrix} \begin{pmatrix} \frac{r-\mu }{\sigma } \end{pmatrix}^{3} \end{bmatrix}
$$
$r$ 代表收益率，$\mu$ 代表均值，$\sigma $ 代表收益率波动率。

偏度在离散数据图像中不容易发现，但是我们可以通过统计来计算。

```python
stats.skew(data_df.PCT_CHG)
# 或者
data_df['PCT_CHG'].skew()
```



##### 四阶中心矩——峰度（Kurtosis）

超额峰度KE公式为：

$$
\large KE=\begin{bmatrix}
\frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum_{i=1}^{n}(\frac{x_{i}-{\mu}}{\sigma })^{4}
\end{bmatrix}-\frac{3(n-1)^{2}}{(n-2)(n-3)}
$$
对于数量很多的样本，超额峰度近似为：
$$
\large KE\approx\frac{1}{n}\sum_{i=1}^{n}(\frac{x_{i}-{\mu}}{\sigma })^{4}-3
$$
一些工具将分布曲线的超额峰度定义为峰度，这样做的目的是让正态分布的峰度重新定义为0，便于分析比较。如Python的Scipy库就是这样处理的。

**`峰度也是对收益率的衡量指标。`**

例如：

$$\large Kurtosis=E\begin{bmatrix}
\begin{pmatrix}
\frac{r-\mu }{\sigma }
\end{pmatrix}^{4}
\end{bmatrix}$$

$r$ 代表收益率，$\mu$ 代表均值，$\sigma $ 代表收益率波动率。

```python
print('收益率的峰度为：',stats.kurtosis(data_df.PCT_CHG))
```







量化应用

**应用一：**

【论文】Conditional volatility, skewness, and kurtosis: existence, persistence, and comovements

【介绍】近些年来，投资组合选择，资产定价和风险价值等模型越来越强调对收益率非对称性和厚尾性建模的重要性。这些特征可以由偏度和峰度表示。在本文之前，大部分学者的研究基于非条件下高阶矩，而对条件矩下密度函数的存在性及其范围并没有做深入的探讨。本文中，作者讨论了广义学生t分布密度函数高阶矩存在时参数的范围并以此作为新息项对GARCH类模型建模。为了检验参数是否具有时变性，作者采用自回归的设定形式，同时考虑了产生虚假结果的可能性。

【链接】<http://www.hec.unil.ch/ejondeau/publications/cJEDC2003a.pdf>

**应用二：**

【研报】偏度和峰度对未来收益率的预测性

【链接】<http://pg.jrj.com.cn/acc/Res/CN_RES/INVEST/2013/5/8/eb385e9d-df79-41c0-9720-ca9fef0ed00d.pdf>





