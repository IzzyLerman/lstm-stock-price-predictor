# lstm-stock-price-predictor

Predicting stock prices one day in advance following the methodology from 'Deep learning framework for stock price prediction using long short-term memory' by S. Kumar Chandar. https://link.springer.com/article/10.1007/s00500-024-09836-3

## Input Features


During preprocessing, 8 technical indicators (TI) are calculated from historical OHLC stock price data. Unless otherwise specified, TIs are calculated using a look-back period of 14 days. These TIs along with the closing price of the stock are used as the features for each example sequence in the training set.

$C_t$--closing price on date $t$, $H_t$--highest price, $L_t$--lowest price, $n$--look-back period

|     Technical Indicator     |      Formula      |
|:---------------------------:|:-----------------:|
|Simple Moving Average (SMA)| $SMA_n=\frac{1}{n}\sum_{i=1}^tC_t$|
|Exponential Moving Average (EMA) | $EMA_n=\frac{2}{1+n}(C_t-EMA_{t-1})+EMA_{t-1}$|
|Relative Strength Index (RSI) | $RSI=100-\frac{100}{1+\frac{AG}{AL}}$ <br>$AG, AL$--Avg. gain/loss over 14 day period|
|Moving Average Convergence Divergence (MACD) | $MACD=EMA_{12}-EMA_{26}$|
|Stochastic %K| $K = 100\frac{C_t-L_1(n)}{H_n(n)-L_1(n)}$ <br> $L_1(n), H_n(n)$ are the highest high and lowest low values observed over the n-day period |
|Stochastic %D| $D = SMA_3(K)$|
|Price Rate of Change (ROC)| $100\frac{C_t}{C_{t-n}}$|
|Williams R%|$100\frac{H_n(n)-C_t}{H_n(n)-L_1(n)}$|





 
