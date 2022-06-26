from re import M
import pandas as pd

PHASE_VARIABLES = ['VOL','M2','_OIL']
def load_data():
    # prepare data
    data_df = pd.read_excel('./market_data.xlsx',sheet_name='data', engine = 'openpyxl')
    data_df.set_index('Date',drop=True,inplace=True)
    return data_df

def correlation_matrix(data, variables):
    matrix = pd.DataFrame({x:data[x] for x in variables})
    return matrix.corr()

if __name__ == '__main__':
    ## Load the data from the sheet
    data = load_data()
    matrix = correlation_matrix(data, PHASE_VARIABLES)

    print(matrix)
    
