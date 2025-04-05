import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
import logging
import json
import os

# 禁用日誌
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('fbprophet').setLevel(logging.WARNING)

# 創建輸出目錄
os.makedirs('output', exist_ok=True)

def save_plotly_fig(fig, filename):
    """保存 Plotly 圖表為 HTML 文件"""
    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    with open(f'output/{filename}', 'w', encoding='utf-8') as f:
        f.write(html)

def create_interactive_plot(ticker_symbol, period):
    try:
        stock = yf.Ticker(ticker_symbol)
        
        if period == "1d":
            interval = "1m"
        elif period == "5d":
            interval = "15m"
        else:
            interval = "1d"
            
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return None
        
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='收盤價',
                line=dict(color='blue'),
                hovertemplate='%{x|%Y-%m-%d %H:%M}<br>價格: %{y:.2f} TWD<extra></extra>'
            ),
            row=1, col=1
        )
        
        if period != "1d":
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist['Open'],
                    name='開盤價',
                    line=dict(color='green', dash='dot'),
                    hovertemplate='%{x|%Y-%m-%d}<br>開盤價: %{y:.2f} TWD<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hist.index.tolist() + hist.index.tolist()[::-1],
                    y=hist['High'].tolist() + hist['Low'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(128,128,128,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    name='高低價範圍'
                ),
                row=1, col=1
            )
        
        fig.update_layout(
            title=f'{ticker_symbol} {period}股價走勢',
            xaxis_title='日期',
            yaxis_title='價格 (TWD)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            height=400
        )
        
        return fig
    except Exception as e:
        print(f"Error creating interactive plot: {str(e)}")
        return None

def create_prophet_forecast(ticker_symbol, periods=30):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None
        
        df = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)
        
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        
        model.add_country_holidays(country_name='TW')
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        fig = plot_plotly(model, forecast)
        fig.update_layout(
            title=f'{ticker_symbol} 未來{periods}天股價預測',
            xaxis_title='日期',
            yaxis_title='價格 (TWD)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            margin=dict(l=50, r=50, b=50, t=80, pad=4),
            height=500
        )
        
        fig.data[0].line.color = 'blue'
        fig.data[1].line.color = 'red'
        fig.data[2].fillcolor = 'rgba(255, 0, 0, 0.2)'
        
        return fig
    except Exception as e:
        print(f"Error creating Prophet forecast: {str(e)}")
        return None

def generate_static_files():
    ticker = "2330.TW"
    tsmc = yf.Ticker(ticker)
    
    # 基本資訊
    info = tsmc.info
    current_price = info.get('currentPrice', 'N/A')
    day_high = info.get('dayHigh', 'N/A')
    day_low = info.get('dayLow', 'N/A')
    previous_close = info.get('previousClose', 'N/A')
    
    # 生成圖表
    plot_1day = create_interactive_plot(ticker, "1d")
    plot_1week = create_interactive_plot(ticker, "5d")
    plot_2weeks = create_interactive_plot(ticker, "10d")
    plot_1month = create_interactive_plot(ticker, "1mo")
    prophet_forecast = create_prophet_forecast(ticker, periods=30)
    
    # 保存圖表
    if plot_1day: save_plotly_fig(plot_1day, "plot_1day.html")
    if plot_1week: save_plotly_fig(plot_1week, "plot_1week.html")
    if plot_2weeks: save_plotly_fig(plot_2weeks, "plot_2weeks.html")
    if plot_1month: save_plotly_fig(plot_1month, "plot_1month.html")
    if prophet_forecast: save_plotly_fig(prophet_forecast, "prophet_forecast.html")
    
    # 歷史數據
    history_data = tsmc.history(period="5d")
    history_list = []
    for date, row in history_data.iterrows():
        history_list.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': round(row['Open'], 2),
            'high': round(row['High'], 2),
            'low': round(row['Low'], 2),
            'close': round(row['Close'], 2),
            'volume': int(row['Volume'])
        })
    
    # 新聞數據
    news = []
    try:
        news_items = tsmc.news[:5]
        for item in news_items:
            news.append({
                'link': item.get('link', '#'),
                'title': item.get('title', '無標題'),
                'publisher': item.get('publisher', '未知來源'),
                'publishedAt': item.get('providerPublishTime', datetime.datetime.now().timestamp())
            })
    except Exception as e:
        print(f"Error processing news: {str(e)}")
    
    # 保存數據為JSON
    data = {
        'current_price': current_price,
        'day_high': day_high,
        'day_low': day_low,
        'previous_close': previous_close,
        'history': history_list,
        'news': news,
        'last_updated': datetime.datetime.now().isoformat()
    }
    
    with open('output/data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    generate_static_files()
