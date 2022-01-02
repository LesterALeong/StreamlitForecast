import streamlit as st
from datetime import datetime, timedelta, date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

from neuralprophet import NeuralProphet, set_log_level

days = (365*10)
START = datetime.now() - timedelta(days=days)
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Asset Forecast App')

stocks = ('BTC-USD','ETH-USD','SPY','^VIX','EURUSD=X',)
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Days of prediction:', 5, 365)
period = n_years


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#m = Prophet()
m = NeuralProphet(num_hidden_layers=96, changepoints_range=0.95, n_changepoints=30, batch_size=64, epochs=10, learning_rate=1.0)
#m.fit(df_train)
#future = m.make_future_dataframe(periods=period)
m.fit(df_train, freq='D')
future = m.make_future_dataframe(df_train, periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} days')
#fig1 = plot_plotly(m, forecast)
fig1 = m.plot(forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)