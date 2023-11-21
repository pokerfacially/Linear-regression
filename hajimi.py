# 加载数据
import pandas as pd
import numpy as np
data = pd.read_csv('demo.csv')
data.head()


#数据散点图展示
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10,10))
fig1 =plt.subplot(231)
plt.scatter(data.loc[:,'Avg. Area Income'],data.loc[:,'Price'])
plt.title('Income')

fig2 =plt.subplot(232)
plt.scatter(data.loc[:,'Avg. Area House Age'],data.loc[:,'Price'])
plt.title('House Age')

fig3 =plt.subplot(233)
plt.scatter(data.loc[:,'Avg. Area Number of Rooms'],data.loc[:,'Price'])
plt.title('Number of Rooms')

fig4 =plt.subplot(234)
plt.scatter(data.loc[:,'Area Population'],data.loc[:,'Price'])
plt.title('Area Population')

fig5 =plt.subplot(235)
plt.scatter(data.loc[:,'size'],data.loc[:,'Price'])
plt.title('size')
plt.show()

#定义 x 和 y
X = data.loc[:,'size']
y = data.loc[:,'Price']
y.head()

# 转换维度
X = np.array(X).reshape(-1,1)
print(X.shape)


#线性回归模型
from sklearn.linear_model import LinearRegression
LR1 = LinearRegression()
#训练模型
LR1.fit(X,y)

#预测
y_predict_1 = LR1.predict(X)
print(y_predict_1)

#模型评估
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error_1 = mean_squared_error(y,y_predict_1)
r2_score_1 = r2_score(y,y_predict_1)
print(mean_squared_error_1,r2_score_1)

# 绘图
fig6 = plt.figure(figsize=(8,5))
plt.scatter(X,y)
plt.plot(X,y_predict_1,'r')
plt.show()

#定义多因子x
#删除掉Price列
X_multi = data.drop(['Price'],axis=1)
X_multi

#第二个线性模型
LR_multi = LinearRegression()
#train the model
LR_multi.fit(X_multi,y)

#多因子预测
y_predict_multi = LR_multi.predict(X_multi)
print(y_predict_multi)

mean_squared_error_multi = mean_squared_error(y,y_predict_multi)
r2_score_multi = r2_score(y,y_predict_multi)
print(mean_squared_error_multi,r2_score_multi)

print(mean_squared_error_1)

fig7 = plt.figure(figsize=(8,5))
plt.scatter(y,y_predict_multi)
plt.show()

fig8 = plt.figure(figsize=(8,5))
plt.scatter(y,y_predict_1)
plt.show()

X_test = [65000,5,5,30000,200]
X_test = np.array(X_test).reshape(1,-1)
print(X_test)

y_test_predict = LR_multi.predict(X_test)
print(y_test_predict)
