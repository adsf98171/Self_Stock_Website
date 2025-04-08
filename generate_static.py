import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
import logging
import json
import sys
from jinja2 import Environment, FileSystemLoader

# 配置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

def fetch_stock_data(ticker_symbol, period, interval=None, retry=3):
    """獲取股票數據，加入重試機制和更強的錯誤處理"""
    for attempt in range(retry):
        try:
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period=period, interval=interval or "1d", timeout=10)
            
            if hist is None or hist.empty:
                logger.warning(f"第 {attempt + 1} 次嘗試: 獲取 {ticker_symbol} 數據為空")
                continue
                
            return hist
            
        except Exception as e:
            logger.error(f"第 {attempt + 1} 次嘗試失敗: {str(e)}")
            if attempt == retry - 1:
                logger.error(f"無法獲取 {ticker_symbol} 數據")
                return pd.DataFrame()  # 返回空 DataFrame 而不是 None
            time.sleep(2)  # 重試間隔

def create_interactive_plot(ticker_symbol, period):
    """創建互動式圖表，完全重寫錯誤處理"""
    try:
        interval = "1m" if period == "1d" else "15m" if period == "5d" else "1d"
        hist = fetch_stock_data(ticker_symbol, period, interval)
        
        if hist.empty:
            logger.warning(f"無法為 {ticker_symbol} 生成 {period} 圖表: 數據為空")
            return "<div class='alert alert-warning'>暫無數據</div>"

        fig = make_subplots(rows=1, cols=1)
        
        # 主價格線
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='收盤價',
                line=dict(color='#1f77b4'),
                hovertemplate='%{x|%Y-%m-%d %H:%M}<br>價格: %{y:.2f} TWD<extra></extra>'
            )
        )
        
        # 非日內數據添加額外資訊
        if period != "1d":
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist['Open'],
                    name='開盤價',
                    line=dict(color='#2ca02c', dash='dot'),
                    opacity=0.7
                )
            )

        fig.update_layout(
            title=f'{ticker_symbol} {period}走勢',
            height=400,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"生成 {period} 圖表失敗: {str(e)}", exc_info=True)
        return "<div class='alert alert-danger'>圖表生成錯誤</div>"

def create_prophet_forecast(ticker_symbol, periods=30):
    """Prophet 預測模型，完全重寫"""
    try:
        hist = fetch_stock_data(ticker_symbol, "1y")
        
        if hist.empty or len(hist) < 30:
            logger.warning("歷史數據不足於30天，無法預測")
            return "<div class='alert alert-warning'>需至少30天數據進行預測</div>"

        # 準備數據
        df = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)
        
        # 訓練模型
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        
        # 添加台灣假期
        model.add_country_holidays(country_name='TW')
        model.fit(df)
        
        # 生成預測
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # 繪製圖表
        # 更新圖表佈局
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
        
        # 將圖表轉為HTML      
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Prophet 預測失敗: {str(e)}", exc_info=True)
        return "<div class='alert alert-danger'>預測系統錯誤</div>"

def generate_static_files():
    """主生成函數"""
    ticker = "2330.TW"
    logger.info(f"開始生成 {ticker} 數據")
    
    # 1. 獲取基本資訊
    try:
        tsmc = yf.Ticker(ticker)
        info = tsmc.info
        current_price = info.get('currentPrice', 'N/A')
        day_high = info.get('dayHigh', 'N/A')
        day_low = info.get('dayLow', 'N/A')
        previous_close = info.get('previousClose', 'N/A')
        logger.info(f"基本資訊獲取成功: 當前價格 {current_price}")
    except Exception as e:
        logger.error(f"獲取基本資訊失敗: {str(e)}")
        current_price = day_high = day_low = previous_close = 'N/A'

    # 2. 生成圖表
    plot_data = {
        'plot_1day': create_interactive_plot(ticker, "1d"),
        'plot_1week': create_interactive_plot(ticker, "5d"),
        'plot_1month': create_interactive_plot(ticker, "1mo"),
        'prophet_forecast': create_prophet_forecast(ticker)
    }

    # 3. 獲取歷史數據
    history_data = fetch_stock_data(ticker, "5d")
    history_list = []
    if not history_data.empty:
        history_list = [{
            'date': datetimeformat(idx),
            'open': round(row['Open'], 2) if 'Open' in row else 'N/A',
            'high': round(row['High'], 2) if 'High' in row else 'N/A',
            'low': round(row['Low'], 2) if 'Low' in row else 'N/A',
            'close': round(row['Close'], 2) if 'Close' in row else 'N/A'
        } for idx, row in history_data.iterrows()]
    else:
        logger.warning("歷史數據為空")

    # 4. 獲取新聞
    news = []
    try:
        if hasattr(tsmc, 'news'):
            news_items = tsmc.news[:5]
            news = [{
                'title': item.get('title', '無標題'),
                'link': item.get('link', '#'),
                'publisher': item.get('publisher', '未知來源'),
                'publishedAt': datetimeformat(datetime.datetime.fromtimestamp(item.get('providerPublishTime', 0)))
            } for item in news_items]
            logger.info(f"獲取 {len(news)} 則新聞")
    except Exception as e:
        logger.error(f"獲取新聞失敗: {str(e)}")

    # 5. 渲染模板
    env = Environment(loader=FileSystemLoader('templates/'))
    env.filters['datetimeformat'] = datetimeformat
    
    try:
        template = env.get_template('generate_static.html')
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
        logger.info("模板渲染成功")
    except Exception as e:
        logger.error(f"模板渲染失敗: {str(e)}")
        html_content = """<h1>系統維護中</h1><p>請稍後再試</p>"""

    # 6. 寫入文件
    try:
        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump({
                'current_price': current_price,
                'history': history_list,
                'news': news,
                'last_updated': datetimeformat(datetime.datetime.now())
            }, f, ensure_ascii=False, indent=2)
        
        logger.info("文件寫入完成")
    except Exception as e:
        logger.error(f"文件寫入失敗: {str(e)}")
        raise  # 重新拋出異常讓 GitHub Actions 捕獲

if __name__ == '__main__':
    try:
        generate_static_files()
    except Exception as e:
        logger.critical(f"程式執行失敗: {str(e)}", exc_info=True)
        sys.exit(1)
