import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import os

# 图像显示中文
font = {'family': 'SimHei'}
plt.rc('font', **font)  # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）


def normalizations(df):
    return df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

def k_means(df):
    max = 0
    index = 0
    df = normalizations(df)
    for i in range(3, 16):
        y_pred = KMeans(n_clusters=i, random_state=9).fit_predict(df)
        r= metrics.calinski_harabaz_score(df, y_pred)
        print(f'k为{i},评分为{r}')
        if r > max:
            max = r
            index = i

    print(f'最优时，k为{index},评分为{max}')

def change_init(df):
    y_pred = KMeans(n_clusters=4, random_state=9,init='k-means++').fit_predict(df)
    r = metrics.calinski_harabaz_score(df, y_pred)
    print(f'中心点方法：k-means++   分数：{r}')
    y_pred = KMeans(n_clusters=4, random_state=9, init='random').fit_predict(df)
    r = metrics.calinski_harabaz_score(df, y_pred)
    print(f'中心点方法：random   分数：{r}')

    # random



if __name__ == '__main__':
    df = pd.read_csv('20181205.csv')
    df=df.set_index('IMSI')
    # print(df['话单数'])

    # df=df.iloc[:,1:]
    # df=df.apply(pd.to_numeric)
    # change_init(df)
    y_pred = KMeans(n_clusters=4, random_state=9, init='k-means++').fit_predict(df)

    types=['话单数', '短信数', '语音时长分钟', '上网流量MB', '上网时长分钟',
           '平均带宽Kbps', '对端号码数', '流量4G占比', '业务线标识', '月份数']
    # plt.figure((10,10))
    for i in types:
        for j in types:
            plt.cla()
            plt.scatter(df[i], df[j], c=y_pred)
            plt.xlabel(i)
            plt.ylabel(j)
            plt.savefig(os.path.join('pic',f'{i}_{j}.png'))
            # plt.show()