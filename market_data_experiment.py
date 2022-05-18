import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

if __name__ == '__main__':
    dr = pd.read_excel('market_data.xlsx', engine='openpyxl')

    # STEP 1: SELECT PAIR of SIGNALS, TARGET
    x_dim = 'Y02'
    y_dim = 'Y10'
    target = '_TY'

    # STEP 2: DISCRETIZE SIGNALS and AVERAGE TARGET for every GRID POINT
    fig, axs = plt.subplots()
    bins = np.linspace(-0.5, 0.5, 30)
    sta, xb, yb, _ = stats.binned_statistic_2d(dr.loc[:,x_dim].diff().values, dr.loc[:,y_dim].diff().values, dr.loc[:,target].values, statistic='mean', bins=bins)
    ps = pd.DataFrame(data=sta, index=["{:.2f}".format(x) for x in xb[1:]], columns=["{:.2f}".format(x) for x in yb[1:]])
    sns.heatmap(ps.iloc[::-1,:], cmap=sns.color_palette("coolwarm", as_cmap=True), vmin = -0.03, vmax = 0.03,  ax=axs)
    axs.set_xlabel(x_dim)
    axs.set_ylabel(y_dim)
    plt.show()

    # STEP 3: STORE OUTPUT (for use with 2d_classifier or auto_encoder)
    with pd.ExcelWriter('experiment_01.xlsx', engine='openpyxl', mode='a') as writer:
        ps.to_excel(writer, sheet_name='2d_classifier')
        ts = (ps.shape[0]*ps.shape[0], 1)  # VECTORIZE 2D PATTERN for use with autoencoder
        pd.DataFrame(ps.values.reshape(ts)).to_excel(writer, sheet_name='auto_encoder')

