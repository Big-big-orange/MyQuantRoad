## 期货策略(套利和海龟)

课程链接：https://www.windquant.com/qntcloud/college







* 本小节自己学到的小技巧

   

  1、均线的表示：

      ind_rec = his_data_df['indicator'][-20:].mean()
      ind_mid = his_data_df['indicator'][-30:].mean()
      ind_lon = his_data_df['indicator'][-40:].mean()

  2、唐安奇通道

  	由3条不同颜色的曲线组成，该指标用周期（一般都是20，有的平台系统设置时可以改变的，有的则设置的不可以）内的最高价和最低价来显	示市场价格的波动性，当其通道窄时表示市场波动较小，反之通道宽则表示市场波动比较大。



  3、talib的简单用法

  ```python
  upperband = ta.MAX(self.high[:-1], timeperiod=self.N2)
  # 首先：s[:-1]的意思就是s字符串取从第0个字符至倒数第一个字符的前一个字符，这样就达到了去掉最后一个字符的目的。   
  # 其次：talib中的MAX函数代表 取周期内最大值（未满足周期返回nan）
  ```

  4、关于期货做空的理解：

  做多就是平时我们买卖股票的模式，先买股票，然后再卖。
  做空就是先卖期货，然后再买回来。通俗点就是，你先借期货卖出去，然后再买回来把他还上就是了。
  一般正规的做空市场是有一个中立仓提供借货的平台。这种模式在价格下跌的波段中能够获利，就是先在高位借货进来卖出，等跌了之后在买进归还






#### 期货跨品种套利

套利也叫价差交易，套利指的是在买入或卖出某种电子交易合约的同时，卖出或买入相关的另一种合约。套利交易是指利用相关市场或相关电子合同之间的价差变化，在相关市场或相关电子合同上进行交易方向相反的交易，以期望价差发生变化而获利的交易行为。

举例：

我们选择豆油、菜籽油、棕榈油这三个品种，历史经验表明他们之间的价格有较高相关性，这很好理解，因为他们之间本身存在一定的替代性，如果某一种油价格相对高了，人们会更倾向于选择价格相对较低的替代品。

用 ((豆油价格-棕榈油价格)-(菜籽油价格-豆油价格))/(菜籽油价格-棕榈油价格) 这个指标来衡量三个品种之间的相对关系。并对这个指标计算其均线，当指标偏离均线时，我们进行配对交易。 交易的方向选为跟随指标的均值回复趋势，即：做空价格高估的品种，同时做多价格低估的品种。



注意：在实际套利中，用什么指标来衡量均衡价差，以及方向选择错误之后的止损策略都是需要仔细斟酌的地方，也是决定套利交易成败的关键之处。

```python
from WindPy import *
import numpy as np
import pandas as pd
from datetime import *
import talib as ta
w.start()

from WindAlgo import * #引入回测框架

## 万矿的回测框架如下：
## wa = BackTest(init_func = initialize, handle_data_func=handle_data)

## 所以要提前定义好initialize ，handle_data
def initialize(context):                            #定义初始化函数
    context.capital = 10000000                      #回测的初始资金
    context.start_date = "20130501"                 #回测开始时间
    context.end_date = "20150701"                  #回测结束时间
    context.benchmark = 'Y.DCE'         #设置豆油为基准
    context.securities = ['Y.DCE','P.DCE','OI.CZC']
    context.period = 'd'                            #策略运行周期, 'd' 代表日, 'm'代表分钟
    context.trade_flag=0
    
def handle_data(bar_datetime, context, bar_data):         #定义策略函数
    trade_flag = context.trade_flag
    count_days = 60
    his_data_df = pd.DataFrame()
    for i in range(len(context.securities)):        
#  history(code, bar_count, period = None, bar_datetime = None)，该函数可用于在回测过程中获取历史行情数据
# 参数说明：
# code: 字符串类型，指定需要获取行情的标的代码，例如'000001.SZ'
# bar_count: 非负整数型，用于指定获取多少个行情周期的数据，例如要获取最近30个回测周期的行情数据，则设置bar_count为30
# 返回值是万矿自定义的dataframe ,包括高开低收，时间和代码名
# period：用于指定周期，默认为回测设置中的context.period 
# get_field 代表获取某一列的数据。
        his_data = wa.history(context.securities[len(context.securities)-i-1], count_days+1)
        temp_data_df = pd.DataFrame(his_data.get_field('close'),index=his_data.get_field('time'),columns=[context.securities[len(context.securities)-i-1]])
        his_data_df = temp_data_df.join(his_data_df)
        # join函数表示按列拼接，并且保留重复列
    
    his_data_df['indicator'] = ((his_data_df['Y.DCE']-his_data_df['P.DCE'])-(his_data_df['OI.CZC']-his_data_df['Y.DCE']))/(his_data_df['OI.CZC']-his_data_df['P.DCE'])
    ind_rec = his_data_df['indicator'][-20:].mean()
    ind_mid = his_data_df['indicator'][-30:].mean()
    ind_lon = his_data_df['indicator'][-40:].mean()
    ind_cur = float(his_data_df['indicator'][-1:])
    
    #交易区
    
    # query_position()，用于当前持仓,返回值是None或者包含订单信息的WindFrame，具体参见帮助文档
    position = wa.query_position()
    if(ind_rec > ind_mid and ind_mid > ind_lon and ind_cur > 1.05*ind_rec):
    # 均线突破，此时豆油价格被高估，棕榈油价格被低估，买入棕榈油，卖出豆油
        if(trade_flag!=1):
            if len(position)>0:
                # 首先卖掉账户中的所有持仓
                res = wa.batch_order.sell_all(price='close', volume_check=False, no_quotation='error')
            # 建立头寸，order_percent买卖的金额等于前一日总资产乘上percent
            res_l = wa.order_percent('P.DCE', 0.1, 'buy',volume_check=True)
            res_s = wa.order_percent('Y.DCE', 0.1, 'short',volume_check=True)
            trade_flag=1
    elif(ind_rec < ind_mid and ind_mid < ind_lon and ind_cur < 0.98*ind_rec):
        if(trade_flag!=-1):
            if len(position)>0:
                res = wa.batch_order.sell_all(price='close', volume_check=False, no_quotation='error')
            
            res_l = wa.order_percent('Y.DCE', 0.1, 'buy',volume_check=True)
            res_s = wa.order_percent('OI.CZC', 0.1, 'short',volume_check=True)
            trade_flag=-1
    elif(ind_cur < 1.02*ind_rec and ind_cur > 0.98*ind_rec):
        if len(position)>0:
            res = wa.batch_order.sell_all(price='close', volume_check=False, no_quotation='error')
        trade_flag=0
        
    context.trade_flag = trade_flag

wa = BackTest(init_func = initialize, handle_data_func=handle_data)          #实例化回测对象
res = wa.run(show_progress=True)                                           #调用run()函数开始回测,show_progress可用于指定是否显示回测净值曲线图
nav_df_hedged = wa.summary('nav') 
```

![1536754553335](Photo\1536754553335.png)







#### 趋势型策略，以海龟策略为例   
策略细节如下：

+ 策略标的：沪深300主力合约


+ 观察周期：日线


+ 交易信号：根据20日的最高、最低、收盘价计算出真实波幅ATR；并计算出近10日的最高与最低价。当价格突破10日最高时多单入场，如果价格继续上涨0.5倍真实波幅则再次加仓；价格回落2倍真实波幅或者突破10日最低则止损止盈出场。开空的思想类似。但是这里需要特别说明的是，无论开仓还是加仓，在海龟系统中，都是根据资金与真实波幅算出买卖的手数。



+ 持仓说明：会根据资金与走势不断加仓，放大盈亏


+ 回测时间：2015年01月01日至2015年11月01日


+ 回测基准：沪深300指数



```python
from WindPy import *
from datetime import *
import pandas as pd
import talib as ta 
w.start()
from WindAlgo import * #引入回测框架

class turtle:
    '''海龟交易系统下的交易信号和仓位管理'''
    def __init__(self, high, low, close, asset, N1=20, N2=10):
        self.high = high
        self.low = low
        self.close = close
        self.asset = asset
        self.N1 = N1
        self.N2 = N2
        
    def trade_signal(self):
        '''唐奇安通道来得到交易信号'''
        upperband = ta.MAX(self.high[:-1], timeperiod=self.N2)
        lowerband = ta.MIN(self.low[:-1], timeperiod=self.N2)
        self.upperlimit = upperband[-1]
        self.lowerlimit = lowerband[-1]
        
    def position(self):
        '''仓位管理——依据波动率水平(ATR)'''
        # TR=（最高价-最低价）和 （最高价-昨收）和（昨收-最低价）的最大值
        vol = ta.ATR(self.high, self.low, self.close, timeperiod=self.N1)
        self.vol = vol[-1]
        unit = max((self.asset * 0.005)/(self.vol*300*0.1),1)   #注意，这里是股指期货，考虑了杠杆？如果换成其他期货品种，请根据品种属性做相应调整
        self.unit = int(unit)     #需要交易的单位数量 —————— 手   
        
def backtest_turtle(stk_code):
    '''海龟回测函数'''
    def initialize(context):                   #定义初始化函数
        context.capital = 1000000              #回测的初始资金
        context.securities = [stk_code]       #回测标的
        context.start_date = "20150101"        #回测开始时间
        context.end_date = "20180110"          #回测结束时间
        context.period = 'd'                   #策略运行周期, 'd' 代表日, 'm'代表分钟
        context.benchmark = stk_code        #策略回测基准
        context.order_id = 0                   #用来记录 加仓或者买入的 订单ID

    def handle_data(bar_datetime, context, bar_data):#定义策略函数

        his = bkt.history(context.securities[0],60)           #使用history函数获取近期历史行情
        high = np.array(his.get_field('high'))
        low = np.array(his.get_field('low'))
        close = np.array(his.get_field('close'))
        asset = bkt.query_capital().get_field('total_asset')[0]   #查看总资产
        turtle_system = turtle(high, low, close, asset)
        turtle_system.trade_signal()
        turtle_system.position()

        position = bkt.query_position()                           #查询当前持仓，期货还能获知是做多单还是空单？
        if context.securities[0] in position.get_field('code'):   # context.securities[0]就是回测标的，这里指的是股指期货
            last_price = bkt.query_order(context.order_id).get_field('price')[0]  #last_price 上一次买入或加仓的价格
            if position.get_field('side')[0] =='long':   # 这里是万得的数据结构，position的返回值有side一项，表示做多还是做空
                if  close[-1] > last_price + 0.5 * turtle_system.vol : # 最近的一次收盘价和ATR信号做比较
                    '''加多单信号'''
                    res_add = bkt.order(context.securities[0], turtle_system.unit, trade_side='buy',price='close',volume_check=False)
                    context.order_id = res_add['order_id']
                if  close[-1] < last_price - 2*turtle_system.vol :     #last_price 上一次买入或加仓的价格
                    '''多单止损'''
                    res = bkt.batch_order.sell_all()        #清仓处理
                    if close[-1] < turtle_system.lowerlimit :
                        res_short = bkt.order(context.securities[0], turtle_system.unit, trade_side='short', price='close',volume_check=False)
                        context.order_id = res_short['order_id']
                if close[-1] < turtle_system.lowerlimit:
                    '''多单止盈'''
                    res = bkt.batch_order.sell_all()
            elif position.get_field('side')[0] =='short':  # 如果持仓属于空单
                if  close[-1] < last_price - 0.5 * turtle_system.vol : 
                    '''加空单信号'''
                    res_add = bkt.order(context.securities[0], turtle_system.unit, trade_side='short',price='close',volume_check=False)
                    context.order_id = res_add['order_id']
                if  close[-1] > last_price + 2*turtle_system.vol :     #last_price 上一次买入或加仓的价格
                    '''空单止损'''
                    res = bkt.batch_order.sell_all()        #清仓处理
                    if close[-1] > turtle_system.upperlimit :
                        res_buy = bkt.order(context.securities[0], turtle_system.unit, trade_side='buy', price='close',volume_check=False)
                        context.order_id = res_buy['order_id']  # 记录交易ID
                if close[-1] > turtle_system.upperlimit:
                    '''空单止盈'''
                    res = bkt.batch_order.sell_all()
        else:   # 如果目前持仓并没有股指期货，那么当突破上轨后，我们就多单买入，突破下轨后就空单买入
            if close[-1] > turtle_system.upperlimit:
                '''多单买入信号'''
                res_buy = bkt.order(context.securities[0], turtle_system.unit, trade_side='buy', price='close',volume_check=False)
                context.order_id = res_buy['order_id']
            elif close[-1] < turtle_system.lowerlimit:
                '''空单买入信号'''
                res_buy = bkt.order(context.securities[0], turtle_system.unit, trade_side='short', price='close',volume_check=False)
                context.order_id = res_buy['order_id']

    bkt = BackTest(init_func = initialize, handle_data_func=handle_data) #实例化回测对象
    res = bkt.run(show_progress=True) #调用run()函数开始回测,show_progress可用于指定是否显示回测净值曲线图
    nav_df=bkt.summary('nav') #获取回测结果

backtest_turtle('IF.CFE')
```

![1536754654193](Photo\1536754654193.png)