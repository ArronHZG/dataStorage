import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 图像显示中文
font = {'family': 'SimHei',
        'weight': 'bold',
        'size': '14'}
plt.rc('font', **font)  # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）


def save_poc(df):
    上市时间 = np.array(df['上市时间'].drop_duplicates())
    上市时间.sort()

    types = '颜色数，上市时间，芯片主频，频段数量，零售价格，厚度，屏幕数量，产品重量，屏幕尺寸，分辩率，RAM，ROM，Flash内存，摄像头，电池容量，文字输入方法数'
    types = types.split('，')
    for type in types:
        data = df[type].groupby(df['上市时间'])
        data_mean = np.array(data.mean())
        plt.cla()
        plt.plot(上市时间, data_mean, color='blue', label=type)
        plt.grid()
        plt.legend()
        plt.xlabel('上市时间')
        plt.ylabel(f'平均{type}')
        plt.savefig(os.path.join('pic', f'{type}.png'))


def print_data_size(df):
    print(df.columns.size)
    print(df.iloc[:, 0].size)


def print_data_mean(df):
    for i in range(3, df.columns.size):
        print(df.iloc[:, i].mean())


def print_data_max(df):
    for i in range(3, df.columns.size):
        print(df.iloc[:, i].max())


def print_data_min(df):
    for i in range(3, df.columns.size):
        print(df.iloc[:, i].min())


def print_data_var(df):
    for i in range(3, df.columns.size):
        print(df.iloc[:, i].var())


def seq_data(df):
    types = '颜色数，上市时间，芯片主频，频段数量，零售价格，厚度，屏幕数量，产品重量，屏幕尺寸，分辩率，RAM，ROM，Flash内存，摄像头，电池容量，文字输入方法数'
    types = types.split('，')
    for type in types:
        print(f'{type} {df[type].max()} {df[type].min()} {df[type].mean()} {df[type].var()}')


def price(df):
    companys = list(np.array(df["品牌"].drop_duplicates()))
    print(companys.__len__())
    data = df['零售价格'].groupby(df['品牌'])
    data_mean = list(np.array(data.mean()))
    print(companys)
    print(data_mean)
    s = []
    for i in range(data_mean.__len__()):
        s.append([companys[i], data_mean[i]])

    print(s)

    mdf = pd.DataFrame(s, columns=['companys', 'price_mean'])
    mdf = mdf.sort_values(by='price_mean')
    print(mdf)
    plt.figure(figsize=(192, 108))

    plt.bar(mdf.iloc[:, 0], mdf.iloc[:, 1], color='blue', label=type)
    # plt.grid()
    # plt.legend()
    plt.xticks(rotation=90, fontsize=30)
    # plt.xticks()
    plt.xlabel('品牌')
    plt.ylabel('零售价格')
    plt.savefig(os.path.join('pic', f'品牌_零售价格.png'))
    # plt.show()


def delect(df):
    df_copy = df
    types = '芯片主频，零售价格，厚度，屏幕数量，产品重量，屏幕尺寸，分辩率，RAM，ROM，Flash内存，摄像头，电池容量'
    types = types.split('，')
    for type in types:
        # df=df_copy
        down = df[type].mean() - 3 * df[type].std()
        up = df[type].mean() + 3 * df[type].std()
        df = df.query(f'{type}<{up} & {type}>{down}')
        print(type, df.iloc[:, 0].size)
        return df


def disperse_isometry(df):
    # 等宽离散
    types = ['芯片主频', '零售价格', '厚度', '屏幕数量', '产品重量',
             '屏幕尺寸', '分辩率', 'RAM', 'ROM', 'Flash内存', '摄像头', '电池容量']
    data = df['芯片主频']
    k = 10
    d = pd.cut(data, k, labels=range(
        k))  # 将回款金额等宽分成k类，命名为0,1,2,3,4,5，data经过cut之后生成了第一列为索引，第二列为当前行的回款金额被划分为0-5的哪一类，属于3这一类的第二列就显示为3

    plt.figure(figsize=(12, 4))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')
    plt.ylim(-0.5, k - 0.5)
    plt.savefig(os.path.join('pic', f'disperse_isometry.png'))
    plt.show()
    # print(df)


def disperse_frequence(df):
    # 等频率离散化
    k = 10
    data = df['芯片主频']
    w = [1.0 * i / k for i in range(k + 1)]
    w = data.describe(percentiles=w)[4:4 + k + 1]
    w[0] = w[0] * (1 - 1e-10)
    w = w.drop_duplicates()
    d = pd.cut(data, w, labels=range(k))

    plt.figure(figsize=(12, 4))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')
    plt.ylim(-0.5, k - 0.5)
    plt.savefig(os.path.join('pic', f'disperse_frequence.png'))
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    # save_poc(df)
    # price(df)
    df = delect(df)
    disperse_frequence(df)
