from synthpop import Synthpop
import pandas as pd
import pyreadr

def synBar():
    df = pd.read_csv("bar_pass_prediction.csv")[['sex', 'race1', 'ugpa', 'bar']]
    print(df.dtypes)
    dtype_map = {'sex': 'float', 'race1': 'category', 'ugpa': 'float', 'bar': 'category'}

    for (k,v) in dtype_map.items():

        if v == 'category':
            df = df.astype({k : "category"})

    print(df.dtypes)
    spop = Synthpop()

    spop.fit(df,dtype_map)

    synth_df = spop.generate(len(df))

    print("synthetische data:")
    print(synth_df.head())

    print("aantal NaNs:")
    print(synth_df.isna().sum())

def synSD2011():
    df0 = pyreadr.read_r("SD2011.rda")['SD2011']
    #pd.read_csv("bar_pass_prediction.csv")
    print(df0.dtypes)
    df = df0[['age', 'unempdur', 'income', 'sex']]#df0[['sex', 'race1', 'ugpa', 'bar']]
    print(df.isna().sum())
    #df.to_excel("inputData.xlsx")
    dtype_map ={
        "age":"float",
        "unempdur":"float",
        "income":"float",
        "sex":"category"
    }
    #{'sex': 'float', 'race1': 'category', 'ugpa': 'float', 'bar': 'category'}
    # for (k,v) in dtype_map.items():
    #     if v == 'category':
    #         df[k] = df[k].astype('category')

    print(df.dtypes)
    r = df.dtypes.keys()
    spop = Synthpop()
    spop.fit(df,dtype_map)

    synth_df = spop.generate(len(df))

    print(synth_df.head())


synSD2011()