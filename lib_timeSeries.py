import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox as ljung
from statsmodels.tsa.stattools import adfuller


### define various error metrics for time series forecasting
def MAE(forecast, observation):
    """ This is a scaled dependent error measure.
        :var forecast: np.ndarray or pd.DataFrame object with the computed forecast
        :var observation: np.ndarray or pd.DataFrame object with the actual observations to compare the forecast to
        :return MAE: mean absolute error (float)
    """
    if isinstance(forecast, pd.DataFrame):
        forecast = np.array(forecast)
    if isinstance(observation, pd.DataFrame):
        observation = np.array(observation)
    err = forecast - observation
    return np.mean(np.abs(err))


def RMSE(forecast, observation):
    """ This is a scaled dependent error measure.
        :var forecast: np.ndarray or pd.DataFrame object with the computed forecast
        :var observation: np.ndarray or pd.DataFrame object with the actual observations to compare the forecast to
        :return RMSE: root mean squared error (float)
    """
    if isinstance(forecast, pd.DataFrame):
        forecast = np.array(forecast)
    if isinstance(observation, pd.DataFrame):
        observation = np.array(observation)
    err = forecast - observation
    return np.sqrt(np.mean(err ** 2))


def MAPE(forecast, observation):
    """ This error measure is a percentage error, thus is unit-free and not scale dependent.
        :var forecast: np.ndarray or pd.DataFrame object with the computed forecast
        :var observation: np.ndarray or pd.DataFrame object with the actual observations to compare the forecast to
        :return: MAPE: mean absolute percentage error (float)
    """
    if isinstance(forecast, pd.DataFrame):
        forecast = np.array(forecast)
    if isinstance(observation, pd.DataFrame):
        observation = np.array(observation)

    err = forecast - observation
    return np.mean(np.abs(err / observation))


def sMAPE(forecast, observation):
    """ This error measure is a percentage error, thus is unit-free and not scale dependent.
        Symmetric mean absolute percentage error: is symmetrizised along forecast and observations
        :var forecast: np.ndarray or pd.DataFrame object with the computed forecast
        :var observation: np.ndarray or pd.DataFrame object with the actual observations to compare the forecast to
        :return: sMAPE: symmetric mean absolute percentage error (float)
    """
    if isinstance(forecast, pd.DataFrame):
        forecast = np.array(forecast)
    if isinstance(observation, pd.DataFrame):
        observation = np.array(observation)
    err = forecast - observation
    return np.mean(2. * np.abs(err) / (forecast + observation))


def MASE(forecast, observation, seasonal=None):
    """ This error measure is a scaled error, thus is not scale dependent.
        The MASE is scaled by the length of the forecast as well as the difference between lags of data
        :var forecast: np.ndarray or pd.DataFrame object with the computed forecast
        :var observation: np.ndarray or pd.DataFrame object with the actual observations to compare the forecast to
        :return: MASE: mean absolute scaled error (float)
    """
    if isinstance(forecast, pd.DataFrame):
        forecast = np.array(forecast)
    if isinstance(observation, pd.DataFrame):
        observation = np.array(observation)
    err = forecast - observation
    T = len(forecast)
    if seasonal == None:
        nlagged = observation[1:]
        lagged = observation[:-1]
        q = err / (1. / (T - 1.) * np.sum(np.abs(nlagged - lagged)))
    else:
        m = seasonal
        nlagged = observation[m + 1:]
        lagged = observation[:-(m + 1)]
        q = err / (1. / (T - m) * np.sum(np.abs(nlagged - lagged)))

    return np.mean(np.abs(q))


### define measurements of strength for trend and seasonality in time series
def measure_trend(data):
    """ This is a measurement of the strength of the trend component in a time series. It was proposed in:
        Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based clustering for time series data.
        Data Mining and Knowledge Discovery, 13(3), 335–364.
        :var data: time series as a pd.DataFrame object (1-dimensional)
        :return: tuple of trend-strength for additive and multiplicative trend components
    """
    assert isinstance(data, pd.DataFrame)

    F = []

    for model in ['additive', 'multiplicative']:
        decomposition = sm.tsa.seasonal_decompose(data[data.columns[:1]], model=model)
        trend = np.array(decomposition.trend.dropna())#.to_numpy()
        resid = np.array(decomposition.resid.dropna())#.to_numpy()
        F_T = max([0, 1. - np.var(np.array(resid)) / np.var(np.array(resid) + np.array(trend))])
        F.append(F_T)
    return (F[0], F[1])


def measure_seasonality(data):
    """ This is a measurement of the strength of the seasonal component in a time series. It was proposed in:
        Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based clustering for time series data.
        Data Mining and Knowledge Discovery, 13(3), 335–364.
        :var data: time series as a pd.DataFrame object (1-dimensional)
        :return: tuple of seasonal-strength for additive and multiplicative seasonal components
    """

    assert isinstance(data, pd.DataFrame)

    F = []

    for model in ['additive', 'multiplicative']:
        decomposition = sm.tsa.seasonal_decompose(data[data.columns[:1]], model=model)
        season = decomposition.seasonal
        resid = decomposition.resid
        ### remove data from seasonal data, where resid has np.nan entries to ensure same lengths
        season = np.array(season[~resid.isna()].dropna())#.to_numpy()
        resid = np.array(resid.dropna())#.to_numpy()
        F_S = max([0, 1. - np.var(np.array(resid)) / np.var(np.array(resid) + np.array(season))])
        F.append(F_S)
    return (F[0], F[1])


def plot_ACF(x, residuals):
    """top figure: plots residuals as a function of the x-coordinate
        left bottom figure: plots histogram with kde
        right bottom figure: plots ACF of residuals
        :var x: is most likely the time variable (np.ndarray)
        :var residuals: already computed residuals of the time series
        """

    ### Minimum and maximum of residuals for y-axis limits
    Min = residuals.min() * 1.2
    Max = residuals.max() * 1.2

    plt.figure(figsize=(16, 8))
    gspec = gs.GridSpec(2, 4)

    top_fig = plt.subplot(gspec[0, :3])
    ### plots residuals as function of x
    top_fig.plot(x, residuals, c='k');
    plt.grid()
    top_fig.set_xlabel("time");
    top_fig.set_ylabel("residuals");
    top_fig.set_ylim(Min, Max);

    top_right = plt.subplot(gspec[0, 3])
    ### plots histogram of residuals beside the residuals plot
    ### shares y-axis with top_fig
    sns.distplot(residuals, rug=True, vertical=True);
    top_right.set_ylim(Min, Max);
    top_right.spines['left'].set_visible(False)
    top_right.yaxis.set_visible(False)
    ### adjust position of figure such that it looks pretty
    pos1 = top_right.get_position()  # get the original position
    pos2 = [pos1.x0 - 0.035, pos1.y0, pos1.width, pos1.height]
    top_right.set_position(pos2)  # set a new position
    ### delete first xtick
    labels = top_right.get_xticks().tolist()
    labels[0] = ''
    top_right.set_xticklabels(labels)

    lags = 20

    left_bottom = plt.subplot(gspec[1, :2])
    ### plots ACF
    left_bottom.set_xlabel("lags");
    # left_bottom.set_ylabel("ACF");
    ### adjust position of figure such that it looks pretty
    pos1 = left_bottom.get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0 - 0.1, pos1.width, pos1.height]
    left_bottom.set_position(pos2)  # set a new position
    left_bottom = plot_acf(residuals, ax=left_bottom, lags=lags, title='ACF', zero=True)

    right_bottom = plt.subplot(gspec[1, 2:])
    ### plots PACF
    right_bottom.set_xlabel("lags");
    # right_bottom.set_ylabel("PACF");
    ### adjust position of figure such that it looks pretty
    pos1 = right_bottom.get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0 - 0.1, pos1.width, pos1.height]
    right_bottom.set_position(pos2)  # set a new position
    right_bottom = plot_pacf(residuals, ax=right_bottom, lags=lags, title='PACF', zero=True)


def ljung_box_test(x, alpha=0.05, seasonal=None):
    """
        Performs ljung-box and box-pierce test on x-data in order to check for auto-correlation
        Remark: still an experimental feature in statsmodels.stats.diagnostic
        :var x: time series input: np.ndarray
        :param alpha: significance level
        :param seasonal: whether the time series has a seasonal component
    """
    if seasonal != None:
        h = min([2 * seasonal, int(len(x) / 5)])
    else:
        h = min([10, int(len(x) / 5)])
    ljung_stat, ljung_p, box_stat, box_p = ljung(x, lags=h, boxpierce=True)

    ### Ljung-Box test
    n = len(x)
    n_ljung = (np.array(ljung_p) < alpha).sum()
    #     EXPERIMENTAL COMPUTE Q VALUE FOR LJUNG BOX TEST
    #     corrs = np.correlate(x, x)[1:]**2
    #     num = np.array(range(len(corrs)))
    #     print(corrs, num)
    #     Q = np.sum(corrs/(n-num))*n*(n+2.)
    #     (chi2, p) = chisquare(x, ddof=0)
    #     print(Q, chi2)
    ### Box-Pierce test
    n_box = (np.array(box_p) < alpha).sum()
    print("Estimated required lag to be h=", h)
    print("Ljung-Box test\n Number of p-values below significance level of " + "{0:.2f}: {1}".format(alpha, n_ljung))
    #     print("Q={0:.2f}>{1:.2f}".format(Q, chi2)) if Q>chi2 else "Q={0:.2f}<{1:.2f}".format(Q, chi2)
    print("Box-Pierce test\n Number of p-values below significance level of " + "{0:.2f}: {1}".format(alpha, n_box))
    print("When there are p-values below the significance level, we can reject the null hypothesis," +
          " meaning that there are still correlations in the data. ")

def aug_dickey_fuller(timeseries, window):
    """
        Test, whether given timeseries is stationary by applying Dickey Fuller test and plot the series
        :param timeseries: pd.DataFrame
        :param window: (int) Size of the moving window. This is the number of observations used for calculating the statistic.
                        Each window will be a fixed size.
    """
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()
#Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xlabel("time index");
    plt.ylabel("scaled value")
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


def plot_decomposition(data, model='additive'):
    """
        Performs a classical decomposition of a time series
        :var data: time series as a pd.DataFrame object (1-dimensional)
        :param model: decomposition type
    """

    assert isinstance(data, pd.DataFrame)
    col = data.columns[0]

    decomposition = sm.tsa.seasonal_decompose(data, model=model)
    season = decomposition.seasonal
    trend = decomposition.trend
    resid = decomposition.resid

    plt.figure(figsize=(16, 8))
    gspec = gs.GridSpec(4, 1)

    ax1 = plt.subplot(gspec[0, 0])
    plt.plot(data.index, data[col]);
    ax1.set_ylabel("data");

    ax2 = plt.subplot(gspec[1, 0])
    plt.plot(data.index, season);
    ax2.set_ylabel("seasonal");
    ### adjust position of figure such that it looks pretty
    pos1 = ax2.get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0 - 0.01, pos1.width, pos1.height]
    ax2.set_position(pos2)  # set a new position

    ax3 = plt.subplot(gspec[2, 0])
    plt.plot(data.index, trend);
    ax3.set_ylabel("trend");
    ### adjust position of figure such that it looks pretty
    pos1 = ax3.get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0 - 0.02, pos1.width, pos1.height]
    ax3.set_position(pos2)  # set a new position

    ax4 = plt.subplot(gspec[3, 0])
    plt.plot(data.index, resid);
    ax4.set_ylabel("remainder");
    ### adjust position of figure such that it looks pretty
    pos1 = ax4.get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0 - 0.03, pos1.width, pos1.height]
    ax4.set_position(pos2)  # set a new position
    ax4.set_xlabel("date");


def forecast_movingaverage(train, test, seasonal=None, my_windowtype = 'hamming'):
    """Computes a forecast for a moving average model
        It uses a naive forecast if there is a seasonal pattern
        and a forecast mean forecast using the last N data points for the remainder
        :var train: training set: pd.DataFrame
        :var test: test set: pd.DataFrame
        :param seasonal: window parameter for rolling mean of seasonal component
        :param my_windowtype: window type used for rolling mean
        :return (predict, observed, std): pd.DataFrames of prediction, observation and standard deviation of training
                                            data compared with moving average (float)
    """

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    col = train.columns[0]

    ### moving averages of training data
    if seasonal == None:
        seasonal = 7
    mavg = train.rolling(window=seasonal, center=True, win_type=my_windowtype).mean()

    ### number of missing data points at end of mavg
    nans_end = int(seasonal / 2 - 1) if seasonal % 2 == 0 else int((seasonal - 1) / 2)
    nans_front = nans_end + 1  # if seasonal % 2 == 0 else nans_end+1
    for n in reversed(range(nans_end + 1)):
        mavg.iloc[-n] = mavg.iloc[-seasonal - n]

    train = train.iloc[nans_front:]
    mavg = mavg.iloc[nans_front:]
    remainder = train - mavg
    std = float(np.std(remainder))

    ### forecasting of test data
    N = 7
    observation, pred = [], []
    hist = list(train[col])
    date = test.index

    for m in range(len(test)):
        season = mavg.iloc[-seasonal + m]
        fluc = np.mean(np.array(remainder.iloc[-1]))
        remainder = remainder.append(pd.DataFrame([fluc], index=[date[m]], columns=[col]))
        forecast = season + fluc

        observation.append(test.iloc[m])
        pred.append(forecast)
        #         hist.append(test.iloc[m])
        hist.append(forecast)
    predict = pd.DataFrame(pred, index=date, columns=[col])
    observed = pd.DataFrame(observation, index=date, columns=[col])
    return (predict, observed, std)

def search_ARIMA_params(data, prange):
    """
    Grid search for optimal ARIMA parameters
    :param data: np.ndarray of input data
    :param prange: int of maximum parameter
    :return: triple of ints with best parameters
    """
    p = q = d = range(0, prange)
    pdq = list(itertools.product(p, d, q))
    params = {}
    for param in pdq:
        try:
            mod = ARIMA(data, order=param)
            results = mod.fit()
            params[param] = results.aic
            print("ARIMA {} - AIC:{}".format(param, results.aic))
        except:
            print("Could not fit model for param ", param)
            continue
    best_param = min(params.items(), key=itemgetter(1))
    print("Found best parameter to be ", best_param[0], "with AIC=", best_param[1])
    return best_param

def search_SARIMA_params(data, prange):
    """
    Grid search for optimal ARIMA parameters
    :param data: np.ndarray of input data
    :param prange: int of maximum parameter
    :return: triple of ints with best parameters
    """
    p = q = d = range(0, prange)
    pdq = list(itertools.product(p, d, q))
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    params = {}
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                params[(param, param_seasonal)] = results.aic
                print('SARIMAX {}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    best_param = min(params.items(), key=itemgetter(1))
    print("Found best parameter to be ", best_param[0], "with AIC=", best_param[1])
    return best_param
