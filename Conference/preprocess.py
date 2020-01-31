import pandas as pd
import os
import matplotlib.pyplot as plt


def read_data():
    os.chdir(os.getcwd() + '/data')
    excel_list = [i for i in os.listdir(os.getcwd()) if i.endswith('.xls')]
    for i, file in enumerate(excel_list, 0):
        tmp_excel = pd.ExcelFile(file, engine='xlrd')
        df = tmp_excel.parse(tmp_excel.sheet_names[1], header=0, index_col='Date')
        df.index = pd.to_datetime(df.index) + pd.to_timedelta(df.Hour, unit='h')
        if i == 0:
            data = df[['DA_DEMD', 'DA_LMP', 'DEMAND', 'RT_LMP']]
        else:
            data = pd.concat([data, df[['DA_DEMD', 'DA_LMP', 'DEMAND', 'RT_LMP']]], axis=0)

    data.to_csv('final_data.csv',index_label='Date')
    return data


# data=read_data()






a=1
