import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
import polars as pl


def ts_analyze(df: pd.DataFrame):

    df = df.sort_values(by=['city', 'timestamp']).reset_index(drop=True)
    
    df['moving_average'] = df.groupby('city')['temperature'].rolling(window=30).mean().reset_index(drop=True)

    df['average_temp_season'] = df.groupby(['city', 'season'])['temperature'].transform('mean')
    df['std_temp_season'] = df.groupby(['city', 'season'])['temperature'].transform('std')

    df['is_anomaly'] = np.abs(df['temperature'] - df['average_temp_season']) > 2 * df['std_temp_season']

    return df


def ts_analyze_pool(data: pd.DataFrame, func, n_cores=4):
    chunk_size = data.shape[0] // n_cores + 1
    
    chunks = [data.iloc[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]

    with Pool(n_cores) as pool:
        results = pool.map(func, chunks)

    return pd.concat(results)


def ts_analyze_polars(df: pl.DataFrame):
    df = df.sort(['city', 'timestamp'])

    df = df.with_columns(
        pl.col('temperature')
        .rolling_mean(window_size=30)
        .over('city')
        .alias('moving_average')
    )

    df = df.with_columns([
        pl.col('temperature')
        .mean()
        .over(['city', 'season'])
        .alias('average_temp_season'),
        
        pl.col('temperature')
        .std()
        .over(['city', 'season'])
        .alias('std_temp_season')
    ])

    df = df.with_columns(
        ((pl.col('temperature') - pl.col('average_temp_season')).abs() > 2 * pl.col('std_temp_season'))
        .alias('is_anomaly')
    )

    return df


if __name__=='__main__':
    data1 = pd.read_csv('temperature_data.csv')
    data2 = data1.copy()
    data3 = data2.copy()
    data4 = data2.copy()

    start_time = time.time()
    df_res = ts_analyze(data1)
    print(f'Время выполнения последовательного алгоритма: {time.time() - start_time} сек')

    start_time = time.time()
    df_res = ts_analyze_pool(data2, func=ts_analyze, n_cores=8)
    print(f'Время выполнения параллельного алгоритма (Pool): {time.time() - start_time} сек')


    data4 = pl.from_pandas(data4)
    start_time = time.time()
    df_res = ts_analyze_polars(data4)
    print(f'Время выполнения параллельного алгоритма (polars): {time.time() - start_time} сек')


'''
Полученные результаты:

Время выполнения последовательного алгоритма: 0.0197751522064209 сек
Время выполнения параллельного алгоритма (Pool): 0.37834906578063965 сек
Время выполнения параллельного алгоритма (polars): 0.0062830448150634766 сек

Выводы:
 Не удалось достичь ускорения при использовании multiprocessing,
 в то время как Polars отрабатывает значительно быстрее при обработке датасета,
 Polars был выбран как метод распараллеливания в приложении на Streamlit
'''