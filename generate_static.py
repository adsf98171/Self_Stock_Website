from flask import Flask, render_template
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
import logging

# 禁用 Prophet 的日誌消息
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('fbprophet').setLevel(logging.WARNING)

app = Flask(__name__)

# 添加日期格式化過濾器
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    try:
        return value.strftime(format)
    except:
        return "N/A"

def create_interactive_plot(ticker_symbol, period):
    try:
        # 獲取股票數據
        stock = yf.Ticker(ticker_symbol)
        
        # 根據不同期間設置適當的interval
        if period == "1d":
            interval = "1m"
        elif period == "5d":
            interval = "15m"
        else:
            interval = "1d"
            
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            raise ValueError("No data available for the given period")
        
        # 創建Plotly圖表
        fig = make_subplots(rows=1, cols=1)
        
        # 添加收盤價線
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
        
        # 如果不是日內數據，添加開盤價和價格區間
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
            
            # 添加高低價範圍
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
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        return plot_html
    except Exception as e:
        print(f"Error creating interactive plot: {str(e)}")
        return None

def create_prophet_forecast(ticker_symbol, periods=30):
    try:
        # 獲取股票數據 (至少1年數據)
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            raise ValueError("No data available for forecasting")
        
        # 準備Prophet數據 - 移除時區信息
        df = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)  # 移除時區信息
        
        # 創建並訓練模型
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        
        # 添加台灣假期
        taiwan_holidays = pd.DataFrame({
            'holiday': 'taiwan_holiday',
            'ds': pd.to_datetime([
                '2025-01-01', '2025-02-28', '2025-04-04', '2025-04-05',
                '2025-05-01', '2025-06-14', '2025-09-21', '2025-10-10'
            ]),
            'lower_window': 0,
            'upper_window': 1,
        })
        model.add_country_holidays(country_name='TW')
        
        model.fit(df)
        
        # 建立未來預測
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # 使用Plotly繪製預測結果
        fig = plot_plotly(model, forecast)
        
        # 自定義圖表佈局
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
        
        # 添加一些自定義樣式
        fig.data[0].line.color = 'blue'  # 實際值
        fig.data[1].line.color = 'red'   # 預測值
        fig.data[2].fillcolor = 'rgba(255, 0, 0, 0.2)'  # 不確定性區間
        
        # 添加模型組件圖
        components_fig = model.plot_components(forecast)
        
        # 將圖表轉為HTML
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        return plot_html
    except Exception as e:
        print(f"Error creating Prophet forecast: {str(e)}")
        return None

@app.route('/')
def tsmc_stock():
    try:
        # 股票代號 (台積電 2330.TW)
        ticker = "2330.TW"
        tsmc = yf.Ticker(ticker)
        
        # 基本資訊
        info = tsmc.info
        current_price = info.get('currentPrice', 'N/A')
        day_high = info.get('dayHigh', 'N/A')
        day_low = info.get('dayLow', 'N/A')
        previous_close = info.get('previousClose', 'N/A')
        
        # 創建互動式圖表
        plot_1day = create_interactive_plot(ticker, "1d")
        plot_1week = create_interactive_plot(ticker, "5d")
        plot_2weeks = create_interactive_plot(ticker, "10d")
        plot_1month = create_interactive_plot(ticker, "1mo")
        
        # 創建Prophet預測圖表
        prophet_forecast = create_prophet_forecast(ticker, periods=30)
        
        # 獲取歷史數據用於表格顯示 (最近5天)
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
        
        # 新聞資料
        news = []
        try:
            news = tsmc.news[:5]  # 只取最近5條新聞
            # 處理新聞數據，確保所有必要的字段都存在
            for item in news:
                item['link'] = item.get('link', '#')
                item['title'] = item.get('title', '無標題')
                item['publisher'] = item.get('publisher', '未知來源')
                item['publishedAt'] = item.get('providerPublishTime', datetime.datetime.now())
        except Exception as e:
            print(f"Error processing news: {str(e)}")
            pass
        
        return render_template('stock_plotly_prophet.html',
                            current_price=current_price,
                            day_high=day_high,
                            day_low=day_low,
                            previous_close=previous_close,
                            plot_1day=plot_1day,
                            plot_1week=plot_1week,
                            plot_2weeks=plot_2weeks,
                            plot_1month=plot_1month,
                            prophet_forecast=prophet_forecast,
                            history_list=history_list,
                            news=news,
                            last_updated=datetime.datetime.now())
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run()
