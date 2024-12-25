# lstm-stock-price-predictor

Predicting stock prices one day in advance following the methodology from 'Deep learning framework for stock price prediction using long short-term memory' by S. Kumar Chandar. https://link.springer.com/article/10.1007/s00500-024-09836-3

## Input Features


During preprocessing, 10 technical indicators (TI) are calculated from historical OHLC stock price data to be used as feature vectors for the LSTM model. TIs are calculated looking back at 14 days of data.

$C_t$--closing price on date $t$, $H_t$--highest price, $L_t$--lowest price

|     Technical Indicator     |      Formula      |
|:---------------------------:|:-----------------:|
|Simple Moving Average (SMA)  | $SMA_n=\frac{1}{n}\sum_{i=1}^tC_t$|
|Enhanced Moving Average (EMA) | $EMA_n=\frac{2}{1+n}(C_t-EMA_{t-1})+EMA_{t-1}$|
|Relative Strength Index (RSI) | $RSI=100-\frac{100}{1+\frac{AG}{AL}}$ <br>$AG, AL$--Avg. gain/loss over 14 day period|
| Moving Average Convergence Divergence (MACD) | $MACD=EMA_{12}-EMA{26}$|




 
