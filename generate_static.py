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

def load_template(template_path):
    """載入 HTML 模板"""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_html(output_path, content):
    """儲存 HTML 檔案"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    """日期格式化函數"""
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
    
    # 新聞資料
    news = []
    try:
        news_items = tsmc.news[:5]
        for item in news_items:
            news.append({
                'link': item.get('link', '#'),
                'title': item.get('title', '無標題'),
                'publisher': item.get('publisher', '未知來源'),
                'publishedAt': datetimeformat(item.get('providerPublishTime', datetime.datetime.now()))
            })
    except Exception as e:
        print(f"Error processing news: {str(e)}")
    
    # 載入模板
    template = load_template('stock_plotly_prophet.html')
    
    # 替換模板中的變數
    html_content = template.replace('{{ current_price }}', str(current_price))
    html_content = html_content.replace('{{ day_high }}', str(day_high))
    html_content = html_content.replace('{{ day_low }}', str(day_low))
    html_content = html_content.replace('{{ previous_close }}', str(previous_close))
    
    # 替換圖表
    html_content = html_content.replace('{{ plot_1day|safe }}', plot_1day or '<p>無法載入今日走勢圖</p>')
    html_content = html_content.replace('{{ plot_1week|safe }}', plot_1week or '<p>無法載入近一週走勢圖</p>')
    html_content = html_content.replace('{{ plot_2weeks|safe }}', plot_2weeks or '<p>無法載入近兩週走勢圖</p>')
    html_content = html_content.replace('{{ plot_1month|safe }}', plot_1month or '<p>無法載入近一個月走勢圖</p>')
    html_content = html_content.replace('{{ prophet_forecast|safe }}', prophet_forecast or '<p>無法載入股價預測圖</p>')
    
    # 替換歷史數據
    history_rows = ""
    for item in history_list:
        history_rows += f"""
        <tr>
            <td>{item['date']}</td>
            <td>{item['open']}</td>
            <td>{item['high']}</td>
            <td>{item['low']}</td>
            <td>{item['close']}</td>
            <td>{item['volume']}</td>
        </tr>
        """
    html_content = html_content.replace('{% for item in history_list %}', '').replace('{% endfor %}', '').replace('{{ item.date }}', '').replace('{{ item.open }}', '').replace('{{ item.high }}', '').replace('{{ item.low }}', '').replace('{{ item.close }}', '').replace('{{ item.volume }}', '')
    html_content = html_content.replace('<tbody>', f'<tbody>{history_rows}')
    
    # 替換新聞數據
    news_items_html = ""
    for item in news:
        news_items_html += f"""
        <div class="news-item">
            <h3><a href="{item['link']}" target="_blank">{item['title']}</a></h3>
            <p>{item['publisher']} - {item['publishedAt']}</p>
        </div>
        """
    html_content = html_content.replace('{% if news %}', '').replace('{% endif %}', '').replace('{% for item in news %}', '').replace('{% endfor %}', '')
    html_content = html_content.replace('<div class="news-container">', f'<div class="news-container">{news_items_html}')
    
    # 替換最後更新時間
    html_content = html_content.replace('{{ last_updated|datetimeformat }}', datetimeformat(datetime.datetime.now()))
    
    # 儲存 HTML 檔案
    save_html('index.html', html_content)
    
    # 儲存數據為 JSON 檔案 (供其他用途)
    data = {
        'current_price': current_price,
        'day_high': day_high,
        'day_low': day_low,
        'previous_close': previous_close,
        'history': history_list,
        'news': news,
        'last_updated': datetime.datetime.now().isoformat()
    }
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    generate_static_files()
