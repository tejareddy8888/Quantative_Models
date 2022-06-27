from re import M
import pandas as pd

from statsmodels.tsa.stattools import grangercausalitytests


from stationaryCheck import StationaryCheck


PHASE_VARIABLES = ['VOL', 'M2', '_OIL', 'CPI', 'Rho']

TARGET_VARIABLES = '_MKT'


def load_data():
    # prepare data
    data_df = pd.read_excel('./market_data.xlsx',
                            sheet_name='data', engine='openpyxl')
    data_df.set_index('Date', drop=True, inplace=True)
    return data_df


def correlation_matrix(data, variables):
    matrix = pd.DataFrame({x: data[x] for x in variables})
    return matrix.corr()


def causationCheck(data, target, phase):
    print("\n Causation results of "+phase+" variable granger causing "+target)
    return grangercausalitytests(data.loc[:, [target, phase]], maxlag=5)


if __name__ == '__main__':
    # Load the data from the sheet
    data = load_data()

    StationaryCheck(data, PHASE_VARIABLES)

    matrix = correlation_matrix(data, PHASE_VARIABLES)

    for eachPhase in PHASE_VARIABLES:
        causationCheck(data, TARGET_VARIABLES, eachPhase)

    print(matrix)
