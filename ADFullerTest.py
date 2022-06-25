from statsmodels.tsa.stattools import adfuller
import pandas as pd

PHASE_VARIABLES = ['Rho','VOL', 'MOV ', 'CF', 'UN', 'GDP', 'M2', 'CPI', 'DIL']

FORECASTING_VARIABLES = ['_MKT', 'MG','RV']

def load_data():
    # prepare data
    data_df = pd.read_excel('./market_data.xlsx',sheet_name='data', engine = 'openpyxl')
    data_df.set_index('Date',drop=True,inplace=True)
    return data_df

def normalize_data(df):
    """ This function can also be called as feature scaling like normalization, min-max scaling, 
    also test sklearn.preprocessing import StandardScaler or other preprocessing
    """
    df = (df - df.mean()) / df.std()
    return df

def StationaryCheck(data,columns):
    for column in columns:
        res = adfuller(data.loc[:, [column]])
        print("_____________"+column+"________________"+"\n")
        # Printing the statistical result of the adfuller test
        print('Augmneted Dickey_fuller Statistic: %f' % res[0])
        print('p-value: %f' % res[1])
        
        # printing the critical values at different alpha levels.
        print('critical values at different levels:')
        for k, v in res[4].items():
            print('\t%s: %.3f' % (k, v))
        print('\n')


if __name__ == '__main__':
    ## Load the data from the sheet
    data = load_data()

    data = normalize_data(data);

    StationaryCheck(data,PHASE_VARIABLES)

    StationaryCheck(data,FORECASTING_VARIABLES)

