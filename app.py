import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import polars as pl
import requests


st.set_page_config(page_title='Отслеживание температуных аномалий', layout="wide")

st.title('Отслеживание температуных аномалий')


# Параллельная функция с использованием polars
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
        .alias('std_temp_season'),

        pl.col('temperature')
        .max()
        .over(['city', 'season'])
        .alias('max_temp_season'),

        pl.col('temperature')
        .min()
        .over(['city', 'season'])
        .alias('min_temp_season')
    ])

    df = df.with_columns(
        ((pl.col('temperature') - pl.col('average_temp_season')).abs() > 2 * pl.col('std_temp_season'))
        .alias('is_anomaly')
    )


    return df



# Отрисовка временного ряда
def plot_ts(df_processed: pd.DataFrame):

    city = df_processed['city'].unique()[0]

    fig = px.line(
        df_processed,
        x='timestamp',
        y='temperature',
        markers='lines+markers',
        color_discrete_sequence=['light blue'],
        labels={
            'temperature': 'Температура, °C',
            'timestamp': 'Дата',
            'is_anomaly': 'Аномальное значение',
        },
        hover_data='is_anomaly',
        title=f'Значения температуры, °C по городу: {city}, исторические данные',
    )

    fig.update_traces(marker=dict(color=['red' if x else 'royalblue' for x in df_processed['is_anomaly']]))
    st.plotly_chart(fig)



# API_KEY = '8422760300a82b441e7e77f164d82625'
def main():
    allowed_extensions = ['.csv', '.xlsx']
    data_file = st.file_uploader('Загрузите файл с историческими данными о погоде',
                     type=allowed_extensions)
    
    if not data_file:
        st.warning('Пожалуйста, загрузите файл')
        return

    file_ext = data_file.name.split('.')[1]
    
    if file_ext == 'csv':
        df = pd.read_csv(data_file)
    else:
        df = pd.read_excel(data_file)
    
    df_pl = pl.from_pandas(df)
    df_pl = ts_analyze_polars(df_pl)
    df = df_pl.to_pandas()
    
    cities = df['city'].unique()

    selected_city = st.selectbox('Выберите город', 
                                 options=cities, placeholder='Выберите город')
    
    API_KEY = st.text_input('Пожалуйста, введите свой OpenWeatherMap API ключ')
    
    df_filtered: pd.DataFrame = df[df['city']==selected_city]

    df_city: pd.DataFrame = df_filtered.groupby('season').agg(
        std_temp_season = ('std_temp_season', 'first'),
        average_temp_season = ('average_temp_season', 'first'),
        min_temp = ('min_temp_season', 'first'),
        max_temp = ('max_temp_season', 'first'),
    )

    df_city = df_city.reset_index().rename(
        columns={
            'season': 'Сезон',
            'std_temp_season': 'Стандартное отклонение за сезон',
            'average_temp_season': 'Средняя температура за сезон',
            'min_temp': 'Минимальная температура за сезон',
            'max_temp': 'Мaксимальная температура за сезон',
        }
    ).set_index('Сезон')

    
    if API_KEY:
        
        url = f"http://api.openweathermap.org/data/2.5/weather?q={selected_city}&appid={API_KEY}&units=metric"
        
        # Использовался модуль requests так как запрос к API всего один
        response = requests.get(url).json()
        
        if response['cod']==401:
            st.warning(response)
            return
        
        cur_temp = response['main']['temp']

        today = pd.Timestamp.now().date()

        if today.month in [1, 2, 12]:
            cur_season = 'winter'
        elif 3<= today.month <= 5:
            cur_season = 'spring'
        elif 6 <= today.month <= 8:
            cur_season = 'summer'
        else:
            cur_season = 'autumn'

        std_season_cur = df_city.loc[cur_season]['Стандартное отклонение за сезон']
        avg_temp_cur_season = df_city.loc[cur_season]['Средняя температура за сезон']
        
        if np.abs(cur_temp - avg_temp_cur_season) > 2 * std_season_cur:
            st.markdown(f'## Температура {today}: {cur_temp} °C. Температура является аномальной для сезона.')
        else:
            st.markdown(f'## Температура {today}: {cur_temp} °C. Температура нормальна для сезона.')
        
    st.write(df_city)

    plot_ts(df_filtered)


if __name__=='__main__':
    main()


    

