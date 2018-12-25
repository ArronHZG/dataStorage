import pandas as pd

data= pd.read_csv("data.csv").values

buys=list()
for item in data:
    buy=list()
    for i in range(10):
        if item[i]==1:
            buy.append(i+1)
    buys.append(buy)

# print(buys)
with open("data1.csv","w+") as f:
    for item in buys:
        line=f"{item[0]}"
        for i in range(1,item.__len__()):
            line+=f",{item[i]}"
        line+='\n'
        f.write(line)