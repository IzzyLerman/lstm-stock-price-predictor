# lstm-stock-price-predictor

Predicting stock prices one day in advance following the methodology from 'Deep learning framework for stock price prediction using long short-term memory' by S. Kumar Chandar. https://link.springer.com/article/10.1007/s00500-024-09836-3

## Input Features


During preprocessing, 10 technical indicators are calculated from historical OHLC stock price data to be used as feature vectors for the LSTM model. 

$C_i$--closing price at time $i$, $H_i$--highest price, $L_i$--lowest price

|     Technical Indicator     |      Formula      |      Description      |
|:---------------------------:|:-----------------:|:----------------------|
|Simple Moving Average (SMA)  | $SMA_t=\frac{1}{t}\sum_{i=1}^tC_i$|Avg. closing price at time $t$|
| 

 
