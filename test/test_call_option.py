# %%
import json
import sqlite3
import pandas as pd 
import numpy as np
from dataframe_to_json import dataframe_to_json

#%%
conn = sqlite3.connect("../db.sqlite3")

cursor = conn.cursor()

cursor.execute("select data from simulation01_parameters where name ='Call Option' ")

row = cursor.fetchall()[0]


    
# 关闭游标和连接
cursor.close()
conn.close()
# %%

row = json.loads(row[0])


#%%
sheet = pd.DataFrame()
data = np.arange(0, 101, 5)
sheet["x"] = data

#%%

row['K']['value']

#%%
#%%




#%%
dataframe_to_json(sheet)
