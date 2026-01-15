import pandas as pd
import numpy as np

df = pd.DataFrame({
    "日期": ["2023-01", "2023-01", "2023-01", "2023-02", "2023-02", "2023-02"],
    "地区": ["北京", "上海", "北京", "上海", "北京", "上海"],
    "产品": ["手机", "手机", "电脑", "手机", "手机", "电脑"],
    "销售额": [1000, 2000, 5000, 1500, 1200, 6000],
    "销量": [1, 2, 1, 2, 1, 1]
})

print(df)

table = pd.pivot_table(df, index="地区", values="销售额", aggfunc='sum')

table = pd.pivot_table(df, index="地区", columns="产品",
                       values="销售额", aggfunc='sum')

print(table)