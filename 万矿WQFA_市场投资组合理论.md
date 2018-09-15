## 市场投资组合理论

课程链接：https://www.windquant.com/qntcloud/college







* 本小节自己学到的小技巧

   

  **numpy 中可以利用numpy.linalg进行多项式拟合**

  **scipy中可以利用from scipy.optimize import leastsq进行最小二乘拟合**

  **sicpy中利用 import scipy.optimize as sco  求解约束最优化问题**







**基础理论：**投资组合是由投资人或金融机构所持有的股票、债券、金融衍生产品等组成的集合。

**核心思想：**资产组合理论的核心思想是资产分散化配置，用以来防范个体风险。

如果按照马科维茨的逻辑，资产配置，就是资产在不同资产产品之间的分配，以求达到方差和期望收益的最佳组合，这个组合的最优解取决于投资者自身的偏好和资本有效配置问题。

资产的配置有效的前提是资产配置位于资产组合的有效边沿上，在此上的资产组合才能根据投资者的具体偏好而做到最优解





### 现代投资组合理论（MPT）

现代资产组合理论（Modern Portfolio Theory，简称MPT），也有人将其称为现代证券投资组合理论、证券组合理论或投资分散理论，

#### 1、均值方差模型

MPT依据以下几个假设：

1、投资者在考虑每一次投资选择时，其依据是某一持仓时间内的证券收益的概率分布。
2、投资者是根据证券的期望收益率估测证券组合的风险。
3、投资者的决定仅仅是依据证券的风险和收益。
4、在一定的风险水平上，投资者期望收益最大；相对应的是在一定的收益水平上，投资者希望风险最小。

根据以上假设，马科维茨确立了证券组合预期收益、风险的计算方法和有效边界理论，建立了资产优化配置的均值－方差模型。

马科维茨把风险定义为期望收益率的波动率，首次将数理统计的方法应用到投资组合选择的研究中。这种模型方法使相互制约的目标能够达到最佳的平衡效果。

一个证券的回报可以表示为服从于正态分布：  

$$
\large\ r\sim N(\mu,\sigma^2)
$$
这里 $\mu$ 表示均值或者期望收益率，$\sigma^2$ 表示方差或者波动率  （收益率=收益/本金） 

用向量 $$\large r= ( \ r_1 ,\ r_2 , \dots ,\ r_n )\ ' $$

表示多个证券的回报，则向量 **r** 服从于多维正态分布 **r**~( **$\mu$**,**$ \sum $**)

其中，**$\mu$** = $(\mu_1, \dots,\mu_n)\ '$ 表示期望收益率向量，**$ \sum $**=$(\sigma_{ij})_{i,j=1}^n$ 表示不同证券收益的协方差矩阵(cov(X,Y)=E[(X-E(X))(Y-E(Y))])



设各品种权重$$\large\omega_1,\omega_2,\dots,\omega_n$$ 

分别表示组合中各证券的权重分布，用向量**$\omega$**=$(\omega_1,\dots,\omega_n) '$ 表示，则证券组合的收益分布可表示为：$$\large \ r_p \sim N(\omega \ ' \mu ,\omega \ ' \sum \omega)$$

**投资组合预期收益率和预期收益的标准差 **

则组合收益率为：$$ \large E(r_p)= \sum w_ir_i $$
组合方差为：
$$
\large  \delta^2(r_p)= \sum\sum w_i w_j cov(r_i,r_j)= w^TVw
$$
$V$ 为协方差矩阵

```python
from WindPy import *
w.start()
import pandas as  pd
import numpy  as  np
import statsmodels.api  as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import scipy.optimize as sco

#取上证50年化收益排名前五的证券作为投资组合
stock=w.wset("sectorconstituent", "date=2017-01-01;sectorId=1000000087000000").Data[1]
startdate='2017-01-01'
enddate='2017-12-31'
#定义收益率计算函数

# 做实证时股票市场的日收益率都用对数收益率（Rt=ln(Pt/Pt-1）) , 而不是等于(Pt-Pt-1)/Pt-1
def ret(stock):
    error_code,data=w.wsd(stock, "close",startdate, enddate, "",usedf=True)
    return np.log(data/data.shift(1)) #采用收盘价，对数收益率计算方法
ret(stock).head()
```

![1536671591078](Photo\1536671591078.png)

```python
# 对上证50所有股票的年化收益率进行排名
r=ret(stock).mean()*252 #求年化收益率 ，年化收益率=日收益率/365*252
r=pd.DataFrame(r,columns=['returns'])
r=r.sort_index(axis = 0,ascending = False,by = 'returns')
r.head(10)
```

![1536671713054](Photo\1536671713054.png)

```python
###下面定义两个函数：

### 1、权重函数weight：用于为组合中的股票随机分配权重  
### 2、投资组合函数portfolio：计算不同权重组合下的期望收益率、方差以及Sharpe比率</br>

# 1、定义随机权重函数
def weight(n):
    w=np.random.random(n)
    return w/sum(w)
weight(5)

# 2、定义投资组合函数 ，给定收益率和各证券权重，计算组和的年化平均收益率、方差以及sharp率
# 采用2017年1年期国债作为无风险资产，年化收益率
# r_f=w.wss("019555.SH", "sec_name,couponrate", "").Data[1][0]/100,无法提取
r_f = 0.04
def portfolio(r,w):
    '''这里的输入r是计算好了的各个投资组合的对数收益率'''
    r_mean=r.mean()*252  #各证券的平均年化收益率
    p_mean=np.sum(r_mean*w)  #组合平均年化收益率
    r_cov=r.cov()*252  #各证券的协方差，这里是pandas的dataframe自带的协方差函数
    p_var=np.dot(w.T, np.dot(r_cov,w))  #计算组合的方差
    p_std=np.sqrt(p_var) #组合标准差
    p_sharpe=(p_mean-r_f)/p_std  # 计算夏普率
    return p_mean,p_std,p_sharpe
portfolio(returns,weight(5))

# 用蒙特卡洛模拟的方法进行不同权重下的投资组合模拟。
p_mean,p_std,p_sharpe=np.column_stack([portfolio(returns,weight(5)) for i in range(8000)]) #产生随机组合
# np.column_stack,一直添加列序列
plt.figure(figsize = (14,8))
plt.scatter(p_std, p_mean, c=p_sharpe, marker = 'o')
plt.grid(True)   #显示网格
plt.xlabel('std')
plt.ylabel('mean')
plt.colorbar(label = 'Sharpe')
plt.title('Mean and Std of Returns')
```

![1536674574665](Photo\1536674574665.png)





#### 2、有效边界(最小方差组合)

有效边界理论亦称有效投资组合理论，是指有效的投资必须满足以下条件之一：  

①同等风险条件下收益最大；  

②同等收益条件下风险最小。  

而有效边界就是给定期望收益风险最小的投资组合集。

**目标函数：**   
$$
\large min\delta^2(r_p)= \sum\sum w_i w_j cov(r_i,r_j)= w^TVw 
$$
**限制条件：**
$$
\large \sum \omega_i =1,\omega_i\geq 0 （不允许卖空）
$$
上式表明，在限制条件下求解 $\omega_i$ 使组合风险 $\delta^2(r_p)$ 最小，可通过拉格朗日目标函数求得:

$$
\large E(\ r_p)=\sum \omega_i \ r_i 
$$

```python
from scipy.optimize import minimize 
t_returns=list(np.linspace(0.5,0.75,100)) #生成100个0.5~0.75下的等步长目标收益率
t_returns[2]
# 任给一个目标收益率(t_returns[2])，求最小标准差
def min_variance(w):
    return portfolio(returns,w)[1]**2  #定义一个最小方差函数，portfolio第二个返回值是标准差
cons = ({'type':'eq','fun':lambda w:portfolio(returns,w)[0]-t_returns[2]},{'type':'eq','fun':lambda w:np.sum(w)-1})

res = minimize(min_variance,weight(5),bounds=((0,1),(0,1),(0,1),(0,1),(0,1)),constraints = cons)

#在不同目标收益率水平（target_returnss)循环时，最小化的一个约束条件会变化
t_std = []
for t in t_returns:
    cons = ({'type':'eq','fun':lambda w:portfolio(returns,w)[0]-t},{'type':'eq','fun':lambda w:np.sum(w)-1})
    res = minimize(min_variance,weight(5),bounds=((0,1),(0,1),(0,1),(0,1),(0,1)),constraints = cons)
    t_std.append(res['fun'])
t_std = np.sqrt(np.array(t_std))    # 这里计算的是标准差


#下面是最优化结果的展示。
# 星号构成的曲线是有效前沿（目标收益率下最优的投资组合）
plt.figure(figsize = (14,8))
#圆圈：蒙特卡洛随机产生的组合分布
plt.scatter(p_std, p_mean, c=p_sharpe, marker = 'o')
#最小方差前沿
plt.scatter(t_std,t_returns, c = (np.array(t_returns)-r_f)/t_std,marker = '*')
plt.grid(True)
plt.xlabel('std')
plt.ylabel('mean')
plt.colorbar(label = 'Sharpe')
plt.title('Mean and Std of Returns')
```

![1536674994237](Photo\1536674994237.png)

```python
#选取最小方差值为起点构建有效前沿
ind=np.argmin(t_std)
e_std = t_std[ind:]
e_returns=t_returns[ind:]
plt.figure(figsize = (14,8))
#圆圈：蒙特卡洛随机产生的组合分布
plt.scatter(p_std, p_mean, c=p_sharpe, marker = 'o')
#最小方差组合
plt.scatter(t_std,t_returns, c = (np.array(t_returns)-r_f)/t_std,marker = '*')
#有效前沿
plt.plot(e_std,e_returns,'r.',markersize=5)
plt.grid(True)
plt.xlabel('std')
plt.ylabel('mean')
plt.colorbar(label = 'Sharpe')
plt.title('Mean and Std of Returns')
```

![1536675049613](Photo\1536675049613.png)

#### 3、Sharp最大时的最优解

```python
# optimize是寻找最小值，这里我们求夏普负值的最小值即为夏普最大值
def min_sharpe(w):
    return -portfolio(returns,w)[2]
cons = ({'type':'eq','fun':lambda w:np.sum(w)-1})
res = minimize(min_sharpe,weight(5),bounds=((0,1),(0,1),(0,1),(0,1),(0,1)),constraints = cons)

#最优组合权重,取小数点后三位
weight=res['x'].round(3)

#sharpe 最大时的组合收益、标准差以及sharpe
p_opt=portfolio(returns,res['x'])

# 星号构成的曲线是有效前沿（目标收益率下最优的投资组合）
plt.figure(figsize = (14,8))
#圆圈：蒙特卡洛随机产生的组合分布
plt.scatter(p_std, p_mean, c=p_sharpe, marker = 'o')
#最小方差组合
plt.scatter(t_std,t_returns, c = (np.array(t_returns)-r_f)/t_std,marker = '*')
#有效前沿
plt.plot(e_std,e_returns,'r.',markersize=5)
#画出Sharpe最大时的点
plt.plot(portfolio(returns,res['x'])[1],portfolio(returns,res['x'])[0],'r*',markersize=20) 
plt.grid(True)
plt.xlabel('std')
plt.ylabel('mean')
plt.colorbar(label = 'Sharpe')
plt.title('Mean and Std of Returns')
```

![1536675295021](Photo\1536675295021.png)





#### 4、资本市场线（CML）

寻找有效边界上切线穿过风险-收益空间上的点(0,r_f)的函数(资本市场线的表达式)：

$$
\large f(x)=a+b*x
$$
 

其中

$$
\large r_f=f(0)=a 
$$

$$
\large b=f'(x)
$$



```python
# 样条插值有两个步骤
# 1、首先要使用splrep()计算欲插值曲线的样条系数（对于N-维空间使用splprep）
# 2、在给定的点上用splev()计算样条插值结果


#使用三次样条插值进行函数的逼近
import scipy.interpolate as sci
tck = sci.splrep(e_std, e_returns,k=3)  # k=3代表3次样条插值，
iy=sci.splev(e_std,tck) #插值估算值

#有效边界函数 (样条函数逼近).
def f(x):
    return sci.splev(x, tck, der=0)   # der是进行样条计算是需要实际计算到的阶数
#有效边界函数f(x)的一阶导数函数
def df(x):
    return sci.splev(x, tck, der=1)
    
    
#定义参数集p=(a,b,x)
def equations(p):
    eq1 = r_f - p[0]
    eq2 = r_f + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3
opt = sco.fsolve(equations, [0.0288, 3.6, 0.18])


#最优化结果带入函数中三个方程均为0，符合预期
np.round(equations(opt),3)

# 星号构成的曲线是有效前沿（目标收益率下最优的投资组合）
plt.figure(figsize = (14,6))
#圆圈：蒙特卡洛随机产生的组合分布
plt.scatter(p_std, p_mean, c=p_sharpe, marker = 'o')
#有效前沿
plt.plot(e_std,e_returns,'r*',markersize=10)
#资本市场线的最优投资组合
plt.plot(opt[2],f(opt[2]),'r*',markersize=20) 
#插值结果
plt.plot(e_std,iy,'g.',markersize=6)
#资本市场线CML
cx=np.linspace(0,0.25)
plt.plot(cx,opt[1]*cx+opt[0],lw=1.5)
plt.grid(True)
plt.xlabel('std')
plt.ylabel('mean')
plt.colorbar(label = 'Sharpe')
plt.title('Mean and Std of Returns')

print("图中蓝色线代表有效边界中切线穿过无风险点的最优化投资组合。这个组合的预期收益率为:",'%.2f%%' % (f(opt[2]) * 100),"预期波动率为:",'%.2f%%' % (opt[2] * 100))
```

![1536752313059](Photo\1536752313059.png)

```python
#最优组合的权重如下
cons = ({'type':'eq','fun':lambda w:(portfolio(returns,w)[0])-f(opt[2])},{'type':'eq','fun':lambda w:np.sum(w)-1}) 
res_cml = minimize(min_variance,weight(5),bounds=((0,1),(0,1),(0,1),(0,1),(0,1)),constraints = cons)
res_cml
```

![1536752393928](Photo\1536752393928.png)

### 资本资产定价模型（CAPM）

资本资产定价模型（Capital Asset Pricing Model 简称CAPM，主要研究证券市场中资产的预期收益率与风险资产之间的关系，以及均衡价格是如何形成的，是现代金融市场价格理论的支柱，广泛应用于投资决策和公司理财领域。  

资本资产定价模型假设所有投资者都按马克维茨的资产选择理论进行投资，对期望收益、方差和协方差等的估计完全相同，投资人可以自由借贷。  

基于这样的假设，资本资产定价模型研究的重点在于探求风险资产收益与风险的数量关系，即为了补偿某一特定程度的风险，投资者应该获得多少的报酬率。



CAPM公式如下：
$$
\large E(\ R_i)-\ R_f=\beta_i(E( \ R_m)- \ R_f)
$$

$$
\large E(\ R_i)= \ R_f +\beta_i(E( \ R_m)- \ R_f)
$$

其中：

$$ E(\ R_i) $$ ：资产期望回报率 

$$\ R_f$$   ：无风险收益率

$$ \beta_i$$  :资产Beta系数，描述资产与市场相关性 

$$
\beta_i = \frac{Cov(\ R_i , \ R_m)}{Var(\ R_m)}\qquad
$$
$$ E(\ R_m)$$   ：市场期望回报率 

$$ E(\ R_m)-\ R_f$$    ：市场风险溢价补偿

$$ E(\ R_i)- \ R_f$$  ：资产溢价补偿



CAPM模型的说明如下：  

1、单个证券的期望收益率由两个部分组成，无风险利率 ($ \ R_f $) 以及对所承担风险的补偿-风险溢价。

2、风险溢价的大小取决于 $ \beta $值的大小。$ \beta $值越高，表明单个证券的风险越高，所得到的补偿也就越高。 

3、$\beta $ 度量的是单个证券的系统风险，非系统性风险没有风险补偿。   



资本资产定价模型（CAPM）实例：

使用CAPM模型和以下假设，我们可以计算股票的预期收益：

无风险率为2%，股票的贝塔（风险度量）为2。在此期间的预期市场回报率为10%，这意味着在从预期市场回报中减去无风险利率之后，市场风险溢价为8%（10%–2%）。将前面的值插入上面的CAPM公式中，我们得到股票的预期收益率为18%：

18%＝2%＋2 x（10%—2%）



```python
# 以上证50为基准市场
error_code,data1=w.wsd("000016.SH", "close",startdate, enddate, "",usedf=True)
SH50_returns=np.log(data1/data1.shift(1)) #采用对数收益率计算方法

base_returns=SH50_returns.mean()*252  #市场的平均年化收益率
base_std=SH50_returns.std()

#根据最优组合权重算组合的序列收益率
weight=res['x'].round(3)
portfolio_returns=np.dot(returns,weight)

d=np.append(portfolio_returns,SH50_returns)
d1=pd.DataFrame(d[1:244],columns=['portfolio'])
d2=pd.DataFrame(d[245:488],columns=['SH50'])
ols_data=pd.merge(d2,d1,left_index=True, right_index=True)

import statsmodels.formula.api as smf
x=ols_data['SH50']
y=ols_data['portfolio']
res = smf.ols(formula='y ~ x', data = ols_data).fit()  # 最小二乘法回归拟合
res.summary()
```

![1536753544332](Photo\1536753544332.png)

```python
X=np.linspace(x.min(), x.max(),100)
Y=res.params[0]+res.params[1]*X
plt.figure(figsize = (10,7.5))
plt.scatter(SH50_returns,portfolio_returns)
plt.plot(X,Y,'r')
plt.title('CAPM Data', fontsize = 20)
plt.xlabel('Returns of SH50', fontsize = 18)
plt.ylabel('Returns of portfolio', fontsize = 18)
plt.grid()
```

![1536753612867](Photo\1536753612867.png)

