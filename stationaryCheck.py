from statsmodels.tsa.stattools import adfuller

import pandas as pd

import pmdarima as pm

from statsmodels.tsa.stattools import grangercausalitytests

INPUT_VARIABLES = ['Rho', 'CPI', 'VOL', 'MOV ', 'CF', 'UN', 'GDP', 'M2',  'DIL', 'M2', '_OIL',]

TARGET_VARIABLES = '_MKT'


def load_data():
    # prepare data
    data_df = pd.read_excel('./market_data.xlsx',
                            sheet_name='data', engine='openpyxl')
    data_df.set_index('Date', drop=True, inplace=True)
    return data_df


def normalize_data(df):
    """ This function can also be called as feature scaling like normalization, min-max scaling, 
    also test sklearn.preprocessing import StandardScaler or other preprocessing
    """
    df = (df - df.mean()) / df.std()
    return df


def StationaryCheck(data, columns):
    p_values = []
    for column in columns:
        res = adfuller(data.loc[:, [column]])
        print("_____________"+column+"________________"+"\n")
        # Printing the statistical result of the adfuller test
        print('Augmneted Dickey_fuller Statistic: %f' % res[0])
        print('p-value: %f' % res[1])

        if res[1] > 0.05:
            print('Variable :'+column+" is not stationary")
        else:
            print('Variable :'+column+" is stationary")

        p_values.append(res[1])

        # printing the critical values at different alpha levels.
        print('critical values at different levels:')
        for k, v in res[4].items():
            print('\t%s: %.3f' % (k, v))
        print('\n')
    return zip(columns, p_values)

def correlation_matrix(data, variables):
    matrix = pd.DataFrame({x: data[x] for x in variables})
    return matrix.corr()


def causationCheck(data, target, phase):
    print("\n Causation results of "+phase+" variable granger causing "+target)
    return grangercausalitytests(data.loc[:, [target, phase]], maxlag=7)

if __name__ == '__main__':
    # Load the data from the sheet
    data = load_data()

    response = StationaryCheck(data, INPUT_VARIABLES)

    for column, p_values in response:
        if p_values > 0.05:
            ndiffs = pm.arima.ndiffs(data[column], alpha=0.05, test='adf', max_d=4)
            print(column+' needs '+str(ndiffs)+' order differentiation')

    matrix = correlation_matrix(data, INPUT_VARIABLES)

    for eachPhase in INPUT_VARIABLES:
        causationCheck(data, TARGET_VARIABLES, eachPhase)

    print(matrix)