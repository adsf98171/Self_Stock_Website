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
import time  # 新增導入time模組用於重試間隔

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
    try:
        # 獲取股票數據 (至少1年數據)
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            raise ValueError("No data available for forecasting")
        
        # 準備Prophet數據 - 移除時區信息
        df = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)

        # 台灣假期設定 (簡化版)
        tw_holidays = pd.DataFrame({
            'holiday': 'tw_market_closed',
            'ds': pd.to_datetime([
                '2024-01-01', '2024-02-28', '2024-04-04', '2024-04-05',
                '2024-05-01', '2024-06-06', '2024-09-17', '2024-10-10',
                '2025-01-01', '2025-02-28', '2025-04-04', '2025-04-05',
                '2025-05-01', '2025-06-07', '2025-09-07', '2025-10-10'
            ]),
            'lower_window': 0,
            'upper_window': 0,
        })
        
        # 創建並訓練模型
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            interval_width=0.95,
            changepoint_prior_scale=0.01, # 降低以避免過擬合初期噪聲
            seasonality_prior_scale=0.1,
            seasonality_mode='multiplicative',
            holidays=tw_holidays,
            holidays_prior_scale=0.5       # 加強事件影響權重
        )
        model.fit(df)
        
        # 建立未來預測 (修正關鍵錯誤)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # 使用Plotly繪製預測結果
        fig = plot_plotly(model, forecast)
        
        # 簡單修改顏色配置
        fig.data[0].line.color = 'black'   # 實際值改為黑色
        fig.data[1].line.color = 'blue'    # 預測值保持藍色
        fig.data[2].fillcolor = 'rgba(0, 255, 0, 0.1)'  # 下半部淡綠色
        fig.data[3].fillcolor = 'rgba(255, 0, 0, 0.1)'  # 上半部淡紅色
        
        # 更新佈局 (保持不變)
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
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"Error creating Prophet forecast: {str(e)}")
        return "<div class='error'>預測生成失敗</div>"

# 在程式開頭定義允許查詢的股票
ALLOWED_STOCKS = {
    '2330.TW': '台積電',
    '2409.TW': '友達',
    '2317.TW': '鴻海',
    # 可以繼續添加其他允許的股票
    '2303.TW': '聯電',
    '1301.TW': '台塑',
    '1303.TW': '南亞'
}

def get_user_input():
    """獲取用戶輸入的股票代號（限制在允許列表內）"""
    print("\n股票數據查詢系統")
    print("=================")
    print("可查詢股票列表:")
    for code, name in ALLOWED_STOCKS.items():
        print(f"{code}: {name}")
    print("\n請輸入股票代號 (例如: 2330.TW)")
    print("輸入 'exit' 或按 Ctrl+C 退出\n")
    
    while True:
        try:
            ticker = input("股票代號: ").strip().upper()
            if ticker.lower() == 'exit':
                return None
            if ticker not in ALLOWED_STOCKS:
                print("錯誤: 不在允許查詢的股票列表中!")
                print("請輸入以下其中一項:", ", ".join(ALLOWED_STOCKS.keys()))
                continue
            return ticker
        except KeyboardInterrupt:
            print("\n程式結束")
            return None
        except Exception as e:
            print(f"輸入錯誤: {str(e)}")
            continue

def generate_stock_report(ticker):
    """生成指定股票的報告"""
    if ticker not in ALLOWED_STOCKS:
        logger.error(f"嘗試查詢未允許的股票: {ticker}")
        return False
    
    logger.info(f"開始生成 {ALLOWED_STOCKS[ticker]}({ticker}) 數據")
    
    # 1. 獲取基本資訊
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
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
        if hasattr(stock, 'news'):
            news_items = stock.news[:5]
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
         # 在渲染模板時傳入 ALLOWED_STOCKS
        html_content = template.render(
            ticker_symbol=ticker,
            stock_name=ALLOWED_STOCKS[ticker],
            ALLOWED_STOCKS=ALLOWED_STOCKS,  # 傳入模板以便顯示
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
        output_filename = f"{ticker.replace('.', '_')}_report.html"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        with open(f"{ticker.replace('.', '_')}_data.json", 'w', encoding='utf-8') as f:
            json.dump({
                'ticker': ticker,
                'current_price': current_price,
                'history': history_list,
                'news': news,
                'last_updated': datetimeformat(datetime.datetime.now())
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文件寫入完成: {output_filename}")
        print(f"\n報告已生成: {output_filename}")
    except Exception as e:
        logger.error(f"文件寫入失敗: {str(e)}")
        raise

def main():
    """主程式入口"""
    while True:
        ticker = get_user_input()
        if not ticker:
            break
        
        try:
            generate_stock_report(ticker)
        except Exception as e:
            print(f"生成報告時發生錯誤: {str(e)}")
            logger.critical(f"生成 {ticker} 報告失敗: {str(e)}", exc_info=True)
        
        print("\n是否要查詢另一支股票? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            break

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"程式執行失敗: {str(e)}", exc_info=True)
        sys.exit(1)
