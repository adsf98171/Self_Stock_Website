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

def save_complete_html(fig, filename, title):
    """生成完整HTML文件（包含Plotly配置）"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ margin: 20px; }}
            .plot-container {{ width: 100%; height: 100%; }}
        </style>
    </head>
    <body>
        <div class="plot-container">
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
    </body>
    </html>
    """
    with open(filename, 'w', encoding='utf-8') as f:
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
    data = {
        'current_price': info.get('currentPrice', 'N/A'),
        'day_high': info.get('dayHigh', 'N/A'),
        'day_low': info.get('dayLow', 'N/A'),
        'previous_close': info.get('previousClose', 'N/A'),
        'last_updated': datetime.datetime.now().isoformat()
    }
    
    # 生成並保存圖表
    periods = {
        '1d': '今日股價',
        '5d': '近5日股價', 
        '10d': '近10日股價',
        '1mo': '近1月股價'
    }
    
    for period, name in periods.items():
        fig = create_interactive_plot(ticker, period)
        if fig:
            save_complete_html(fig, f"plot_{period}.html", f"台積電({ticker}) {name}走勢")
    
    # 保存Prophet預測
    prophet_fig = create_prophet_forecast(ticker)
    if prophet_fig:
        save_complete_html(prophet_fig, "prophet_forecast.html", "台積電(2330.TW) 未來30天股價預測")
    
    # 保存歷史數據
    history_data = tsmc.history(period="5d")
    data['history'] = [{
        'date': date.strftime('%Y-%m-%d'),
        'open': round(row['Open'], 2),
        'high': round(row['High'], 2),
        'low': round(row['Low'], 2),
        'close': round(row['Close'], 2),
        'volume': int(row['Volume'])
    } for date, row in history_data.iterrows()]
    
    # 保存新聞數據
    try:
        data['news'] = [{
            'link': item.get('link', '#'),
            'title': item.get('title', '無標題'),
            'publisher': item.get('publisher', '未知來源'),
            'publishedAt': item.get('providerPublishTime', datetime.datetime.now().timestamp())
        } for item in tsmc.news[:5]]
    except Exception as e:
        print(f"Error processing news: {str(e)}")
        data['news'] = []
    
    # 保存數據
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    generate_static_files()
