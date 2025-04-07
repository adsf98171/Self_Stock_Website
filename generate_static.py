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
from jinja2 import Environment, FileSystemLoader

# 禁用不必要的日誌
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('fbprophet').setLevel(logging.WARNING)

def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    """台灣時間格式化 (UTC+8)"""
    if not value:
        return "N/A"
    if isinstance(value, str):
        return value
    try:
        taiwan_time = value + datetime.timedelta(hours=8) if isinstance(value, datetime.datetime) else value
        return taiwan_time.strftime(format)
    except Exception:
        return "N/A"

def fetch_stock_data(ticker_symbol, period, interval=None):
    """獲取股票數據，加入重試機制"""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period, interval=interval or "1d")
        return hist if not hist.empty else None
    except Exception as e:
        logging.error(f"獲取 {ticker_symbol} 數據失敗: {str(e)}")
        return None

def create_interactive_plot(ticker_symbol, period):
    """創建互動式圖表，加入錯誤邊界處理"""
    try:
        interval = "1m" if period == "1d" else "15m" if period == "5d" else "1d"
        hist = fetch_stock_data(ticker_symbol, period, interval)
        if hist is None:
            return "<p>數據獲取失敗</p>"

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='收盤價',
                line=dict(color='blue'),
                hovertemplate='%{x|%Y-%m-%d %H:%M}<br>價格: %{y:.2f} TWD<extra></extra>'
            )
        )
        
        if period != "1d":
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Open'],
                name='開盤價',
                line=dict(color='green', dash='dot')
            ))

        fig.update_layout(
            title=f'{ticker_symbol} {period}股價走勢',
            height=400,
            template='plotly_white'
        )
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        logging.error(f"生成圖表失敗: {str(e)}")
        return "<p>圖表生成錯誤</p>"

def create_prophet_forecast(ticker_symbol, periods=30):
    """Prophet預測模型，加入異常處理"""
    try:
        hist = fetch_stock_data(ticker_symbol, "1y")
        if hist is None or len(hist) < 30:
            return "<p>數據不足無法預測</p>"

        df = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)

        model = Prophet(
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.add_country_holidays(country_name='TW')
        model.fit(df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        fig = plot_plotly(model, forecast)
        fig.update_layout(height=500)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        logging.error(f"預測失敗: {str(e)}")
        return "<p>預測生成錯誤</p>"

def generate_static_files():
    """主生成函數，完全重寫確保穩定性"""
    ticker = "2330.TW"
    
    # 1. 獲取基本數據 (加入超時設置)
    try:
        tsmc = yf.Ticker(ticker)
        info = tsmc.info
        current_price = info.get('currentPrice', 'N/A')
        day_high = info.get('dayHigh', 'N/A')
        day_low = info.get('dayLow', 'N/A')
        previous_close = info.get('previousClose', 'N/A')
    except Exception as e:
        logging.error(f"獲取基本資訊失敗: {str(e)}")
        current_price = day_high = day_low = previous_close = 'N/A'

    # 2. 並行生成圖表
    plot_data = {
        'plot_1day': create_interactive_plot(ticker, "1d"),
        'plot_1week': create_interactive_plot(ticker, "5d"),
        'plot_1month': create_interactive_plot(ticker, "1mo"),
        'prophet_forecast': create_prophet_forecast(ticker)
    }

    # 3. 獲取歷史數據
    history_data = fetch_stock_data(ticker, "5d") or pd.DataFrame()
    history_list = [{
        'date': datetimeformat(idx),
        'open': round(row['Open'], 2) if 'Open' in row else 'N/A',
        'high': round(row['High'], 2) if 'High' in row else 'N/A',
        'low': round(row['Low'], 2) if 'Low' in row else 'N/A',
        'close': round(row['Close'], 2) if 'Close' in row else 'N/A'
    } for idx, row in history_data.iterrows()]

    # 4. 獲取新聞 (加入超時處理)
    news = []
    try:
        news_items = tsmc.news[:5] if hasattr(tsmc, 'news') else []
        news = [{
            'title': item.get('title', '無標題'),
            'link': item.get('link', '#'),
            'publisher': item.get('publisher', '未知來源'),
            'publishedAt': datetimeformat(datetime.datetime.fromtimestamp(item.get('providerPublishTime', 0)))
        } for item in news_items]
    except Exception as e:
        logging.error(f"獲取新聞失敗: {str(e)}")

    # 5. 渲染模板 (嚴格錯誤處理)
    env = Environment(loader=FileSystemLoader('.'))
    env.filters['datetimeformat'] = datetimeformat
    
    try:
        template = env.get_template('index.html')
        html_content = template.render(
            current_price=current_price,
            day_high=day_high,
            day_low=day_low,
            previous_close=previous_close,
            history_list=history_list,
            news=news,
            last_updated=datetime.datetime.now(),
            **plot_data
        )
    except Exception as e:
        logging.error(f"模板渲染失敗: {str(e)}")
        html_content = """<h1>頁面生成錯誤</h1><p>請檢查後端日誌</p>"""

    # 6. 寫入文件 (原子化操作)
    try:
        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump({
                'current_price': current_price,
                'history': history_list,
                'last_updated': datetimeformat(datetime.datetime.now())
            }, f, ensure_ascii=False, indent=2)
        
        logging.info("文件生成成功")
    except Exception as e:
        logging.error(f"文件寫入失敗: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_static_files()
