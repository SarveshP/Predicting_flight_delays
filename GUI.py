from flask import Flask
from flask import abort
from flask import request
from flask import jsonify

from flask import session
from flask import g
from flask import redirect
from flask import url_for
from flask import render_template
from flask import flash
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import os
import datetime, warnings, scipy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from random import sample
import matplotlib.patches as mpatches
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

app = Flask(__name__)
Bootstrap(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

target = os.path.join(APP_ROOT, 'files/')


def default_datasets(carrier, id_airport):
    # # **Predicting flight delays**

    # In this notebook, we developed the model aimed at predicting flight delays at take-off.

    # During the EDA, we intended to create good quality figures

    # This notebook is composed of three parts:
    # Cleaning
    #   *  Date and Times
    #   *  Missing Values

    # Exploration
    #   * Graphs
    #   * Impact of Departure Vs Arrival Delays

    # Modeling
    # The model is developed for one airport and one airline
    #   * Linear
    #   * Ridge
    #   * Random Forest
    #   * Neural Networks
    #   * SVM


    # In[2]:


    import datetime, warnings, scipy
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import ConnectionPatch
    from collections import OrderedDict
    from matplotlib.gridspec import GridSpec
    from sklearn import metrics, linear_model
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    from random import sample
    import matplotlib.patches as mpatches
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr, pearsonr
    from sklearn.svm import SVR
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    mpl.rc('patch', edgecolor='dimgray', linewidth=1)
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "last_expr"
    pd.options.display.max_columns = 50
    #get_ipython().magic('matplotlib inline')
    warnings.filterwarnings("ignore")

    # In[2]:


    df = pd.read_csv('/Users/sarveshprattipati/Downloads/flight-delays/flights.csv', low_memory=False)
    print('Dataframe dimensions:', df.shape)

    airports = pd.read_csv("/Users/sarveshprattipati/Downloads/flight-delays/airports.csv")

    airlines_names = pd.read_csv('/Users/sarveshprattipati/Downloads/flight-delays/airlines.csv')
    airlines_names

    abbr_companies = airlines_names.set_index('IATA_CODE')['AIRLINE'].to_dict()

    carrier = 'AA'
    id_airport = 'DFW'

    # %%

    # # 1. Cleaning

    # # 1.1 Dates and times
    #
    # **YEAR, MONTH, DAY**, is merged into date column

    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

    # Moreover, in the **SCHEDULED_DEPARTURE** variable, the hour of the take-off is coded as a float where the two first digits indicate the hour and the two last, the minutes. This format is not convenient and I thus convert it. Finally, I merge the take-off hour with the flight date. To proceed with these transformations, I define a few functions:

    # Function that converts the 'HHMM' string to datetime.time
    def format_heure(chaine):
        if pd.isnull(chaine):
            return np.nan
        else:
            if chaine == 2400: chaine = 0
            chaine = "{0:04d}".format(int(chaine))
            heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
            return heure

    # Function that combines a date and time to produce a datetime.datetime
    def combine_date_heure(x):
        if pd.isnull(x[0]) or pd.isnull(x[1]):
            return np.nan
        else:
            return datetime.datetime.combine(x[0], x[1])

    # Function that combine two columns of the dataframe to create a datetime format
    def create_flight_time(df, col):
        liste = []
        for index, cols in df[['DATE', col]].iterrows():
            if pd.isnull(cols[1]):
                liste.append(np.nan)
            elif float(cols[1]) == 2400:
                cols[0] += datetime.timedelta(days=1)
                cols[1] = datetime.time(0, 0)
                liste.append(combine_date_heure(cols))
            else:
                cols[1] = format_heure(cols[1])
                liste.append(combine_date_heure(cols))
        return pd.Series(liste)

    df['SCHEDULED_DEPARTURE'] = create_flight_time(df, 'SCHEDULED_DEPARTURE')
    df['DEPARTURE_TIME'] = df['DEPARTURE_TIME'].apply(format_heure)
    df['SCHEDULED_ARRIVAL'] = df['SCHEDULED_ARRIVAL'].apply(format_heure)
    df['ARRIVAL_TIME'] = df['ARRIVAL_TIME'].apply(format_heure)
    # __________________________________________________________________________
    # df.loc[:5, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',
    #             'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']]


    # The content of the **DEPARTURE_TIME** and **ARRIVAL_TIME** variables can be a bit misleading.
    # the first entry of the dataframe, the scheduled departure is at 0h05 the 1st of January.
    # ### 1.2 Filling factor
    #
    # Finally, the data frame is cleaned and few columns are dropped
    variables_to_remove = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR',
                           'MONTH', 'DAY', 'DAY_OF_WEEK', 'DATE', 'AIR_SYSTEM_DELAY',
                           'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
                           'WEATHER_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
                           'FLIGHT_NUMBER', 'TAIL_NUMBER', 'AIR_TIME']
    df.drop(variables_to_remove, axis=1, inplace=True)
    df = df[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
             'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY',
             'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
             'SCHEDULED_TIME', 'ELAPSED_TIME']]
    # df[:5]

    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['variable', 'missing values']
    missing_df['filling factor (%)'] = (df.shape[0] - missing_df['missing values']) / df.shape[0] * 100
    missing_df.sort_values('filling factor (%)').reset_index(drop=True)

    # The filling factor is quite good (> 97%). So dropping the rows with NA is a good option
    df.dropna(inplace=True)

    # %%
    # # 2. Exploration
    # # 2.1 Basic statistical description of airlines

    # function for statistical parameters from a grouby object:
    def get_stats(group):
        return {'min': group.min(), 'max': group.max(),
                'count': group.count(), 'mean': group.mean()}

    global_stats = df['DEPARTURE_DELAY'].groupby(df['AIRLINE']).apply(get_stats).unstack()
    global_stats = global_stats.sort_values('count')
    global_stats

    # In[15]:

    # # 2.1 Graphs

    # Pie chart for

    font = {'family': 'normal', 'weight': 'bold', 'size': 15}
    mpl.rc('font', **font)

    # __________________________________________________________________
    # I extract a subset of columns and redefine the airlines labeling
    df2 = df.loc[:, ['AIRLINE', 'DEPARTURE_DELAY']]
    df2['AIRLINE'] = df2['AIRLINE'].replace(abbr_companies)
    # ________________________________________________________________________
    colors = ['royalblue', 'grey', 'wheat', 'c', 'firebrick', 'seagreen', 'lightskyblue',
              'lightcoral', 'yellowgreen', 'gold', 'tomato', 'violet', 'aquamarine', 'chartreuse']
    # ___________________________________
    fig = plt.figure(1, figsize=(16, 15))
    gs = GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    labels = [s for s in global_stats.index]
    # ----------------------------------------
    # Pie chart for mean delay at departure
    # ----------------------------------------
    sizes = global_stats['mean'].values
    sizes = [max(s, 0) for s in sizes]
    explode = [0.0 if sizes[i] < 20000 else 0.01 for i in range(len(abbr_companies))]
    patches, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                        colors=colors, shadow=False, startangle=0,
                                        autopct=lambda p: '{:.0f}'.format(p * sum(sizes) / 100))
    for i in range(len(abbr_companies)):
        texts[i].set_fontsize(14)
    ax1.axis('equal')
    ax1.set_title('Mean delay at origin', bbox={'facecolor': 'midnightblue', 'pad': 5},
                  color='w', fontsize=18)
    # ------------------------------------------------------
    # striplot with all the values for the delays
    # ___________________________________________________________________
    # Defining the colors for correspondance with the pie charts
    colors = ['firebrick', 'gold', 'lightcoral', 'aquamarine', 'c', 'yellowgreen', 'grey',
              'seagreen', 'tomato', 'violet', 'wheat', 'chartreuse', 'lightskyblue', 'royalblue']
    # ___________________________________________________________________
    ax2 = sns.stripplot(y="AIRLINE", x="DEPARTURE_DELAY", size=4, palette=colors,
                        data=df2, linewidth=0.5, jitter=True)
    plt.setp(ax2.get_xticklabels(), fontsize=14)
    plt.setp(ax2.get_yticklabels(), fontsize=14)
    ax2.set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*[int(y) for y in divmod(x, 60)])
                         for x in ax2.get_xticks()])
    plt.xlabel('Departure delay', fontsize=18, bbox={'facecolor': 'midnightblue', 'pad': 5},
               color='w', labelpad=20)
    ax2.yaxis.label.set_visible(False)
    # ________________________
    plt.tight_layout(w_pad=3)

    # If we Exclude Hawaiian Airlines and Alaska Airlines, which have low mean delays, the mean delay would be 11 ± 7 minutes
    # The second graph shows that, incase of mean delay being 11 minutes, there might be hours delay for some flights

    # In[16]:

    # # 2.1 Graphs

    # Function defining how delays are grouped
    delay_type = lambda x: ((0, 1)[x > 5], 2)[x > 45]
    df['DELAY_LEVEL'] = df['DEPARTURE_DELAY'].apply(delay_type)

    fig = plt.figure(1, figsize=(10, 7))
    ax = sns.countplot(y="AIRLINE", hue='DELAY_LEVEL', data=df)

    # We replace the abbreviations by the full names of the companies and set the labels
    labels = [abbr_companies[item.get_text()] for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), fontsize=12, weight='normal', rotation=0);
    plt.setp(ax.get_yticklabels(), fontsize=12, weight='bold', rotation=0);
    ax.yaxis.label.set_visible(False)
    plt.xlabel('Flight count', fontsize=16, weight='bold', labelpad=10)

    # Set the legend
    L = plt.legend()
    L.get_texts()[0].set_text('on time (t < 5 min)')
    L.get_texts()[1].set_text('small delay (5 < t < 45 min)')
    L.get_texts()[2].set_text('large delay (t > 45 min)')
    plt.show()

    # %%

    # # 2.2 Impact of Departure Vs Arrival Delays

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['hatch.linewidth'] = 2.0

    fig = plt.figure(1, figsize=(11, 6))
    ax = sns.barplot(x="DEPARTURE_DELAY", y="AIRLINE", data=df, color="lightskyblue", ci=None)
    ax = sns.barplot(x="ARRIVAL_DELAY", y="AIRLINE", data=df, color="r", hatch='///',
                     alpha=0.0, ci=None)
    labels = [abbr_companies[item.get_text()] for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    ax.yaxis.label.set_visible(False)
    plt.xlabel('Mean delay [min] (@departure: blue, @arrival: hatch lines)',
               fontsize=14, weight='bold', labelpad=10);

    # This figure shows arrival delays are lower than departure delays.
    # The arrival delays can be compensated during air travel.

    # So for this project we have estimating the departure delays.


    # %%

    # ### 2.2 Vizualization for delays at origin airports

    airport_mean_delays = pd.DataFrame(pd.Series(df['ORIGIN_AIRPORT'].unique()))
    airport_mean_delays.set_index(0, drop=True, inplace=True)

    for carrier in abbr_companies.keys():
        df1 = df[df['AIRLINE'] == carrier]
        test = df1['DEPARTURE_DELAY'].groupby(df['ORIGIN_AIRPORT']).apply(get_stats).unstack()
        airport_mean_delays[carrier] = test.loc[:, 'mean']

    temp_airports = airports
    identify_airport = temp_airports.set_index('IATA_CODE')['CITY'].to_dict()

    sns.set(context="paper")
    fig = plt.figure(1, figsize=(8, 8))

    ax = fig.add_subplot(1, 2, 1)
    subset = airport_mean_delays.iloc[:50, :].rename(columns=abbr_companies)
    subset = subset.rename(index=identify_airport)
    mask = subset.isnull()
    sns.heatmap(subset, linewidths=0.01, cmap="Accent", mask=mask, vmin=0, vmax=35)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation=85);
    ax.yaxis.label.set_visible(False)

    ax = fig.add_subplot(1, 2, 2)
    subset = airport_mean_delays.iloc[50:100, :].rename(columns=abbr_companies)
    subset = subset.rename(index=identify_airport)
    fig.text(0.5, 1.02, "Delays: impact of the origin airport", ha='center', fontsize=18)
    mask = subset.isnull()
    sns.heatmap(subset, linewidths=0.01, cmap="Accent", mask=mask, vmin=0, vmax=35)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation=85);
    ax.yaxis.label.set_visible(False)

    plt.tight_layout()

    # From the above graph, we deduce
    # American eagle has large delays
    # Delta airlines has delays less than 5 minutes
    # Few airports favour late departure,like Denver, Chicago

    # In[32]:

    # Common class for graphs
    class Figure_style():
        # _________________________________________________________________
        def __init__(self, size_x=11, size_y=5, nrows=1, ncols=1):
            sns.set_style("white")
            sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
            self.fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size_x, size_y,))
            # ________________________________
            # convert self.axs to 2D array
            if nrows == 1 and ncols == 1:
                self.axs = np.reshape(axs, (1, -1))
            elif nrows == 1:
                self.axs = np.reshape(axs, (1, -1))
            elif ncols == 1:
                self.axs = np.reshape(axs, (-1, 1))

        # _____________________________
        def pos_update(self, ix, iy):
            self.ix, self.iy = ix, iy

        # _______________
        def style(self):
            self.axs[self.ix, self.iy].spines['right'].set_visible(False)
            self.axs[self.ix, self.iy].spines['top'].set_visible(False)
            self.axs[self.ix, self.iy].yaxis.grid(color='lightgray', linestyle=':')
            self.axs[self.ix, self.iy].xaxis.grid(color='lightgray', linestyle=':')
            self.axs[self.ix, self.iy].tick_params(axis='both', which='major',
                                                   labelsize=10, size=5)

        # ________________________________________
        def draw_legend(self, location='upper right'):
            legend = self.axs[self.ix, self.iy].legend(loc=location, shadow=True,
                                                       facecolor='g', frameon=True)
            legend.get_frame().set_facecolor('whitesmoke')

        # _________________________________________________________________________________
        def cust_plot(self, x, y, color='b', linestyle='-', linewidth=1, marker=None, label=''):
            if marker:
                markerfacecolor, marker, markersize = marker[:]
                self.axs[self.ix, self.iy].plot(x, y, color=color, linestyle=linestyle,
                                                linewidth=linewidth, marker=marker, label=label,
                                                markerfacecolor=markerfacecolor, markersize=markersize)
            else:
                self.axs[self.ix, self.iy].plot(x, y, color=color, linestyle=linestyle,
                                                linewidth=linewidth, label=label)
            self.fig.autofmt_xdate()

        # ________________________________________________________________________
        def cust_plot_date(self, x, y, color='lightblue', linestyle='-',
                           linewidth=1, markeredge=False, label=''):
            markeredgewidth = 1 if markeredge else 0
            self.axs[self.ix, self.iy].plot_date(x, y, color='lightblue', markeredgecolor='grey',
                                                 markeredgewidth=markeredgewidth, label=label)

        # ________________________________________________________________________
        def cust_scatter(self, x, y, color='lightblue', markeredge=False, label=''):
            markeredgewidth = 1 if markeredge else 0
            self.axs[self.ix, self.iy].scatter(x, y, color=color, edgecolor='grey',
                                               linewidths=markeredgewidth, label=label)
            #

        def set_xlabel(self, label, fontsize=14):
            self.axs[self.ix, self.iy].set_xlabel(label, fontsize=fontsize)

        def set_ylabel(self, label, fontsize=14):
            self.axs[self.ix, self.iy].set_ylabel(label, fontsize=fontsize)

        # ____________________________________
        def set_xlim(self, lim_inf, lim_sup):
            self.axs[self.ix, self.iy].set_xlim([lim_inf, lim_sup])

        # ____________________________________
        def set_ylim(self, lim_inf, lim_sup):
            self.axs[self.ix, self.iy].set_ylim([lim_inf, lim_sup])


    # Sampling the data with 80:20 training and test data set
    df_train = df.sample(frac=0.8)
    df_test = df.loc[~df.index.isin(df_train.index)]
    df = df_train

    # In[37]:
    # Defining dataframe creation function
    ###########################################################################
    def get_flight_delays(df, carrier, id_airport, extrem_values=False):
        df2 = df[(df['AIRLINE'] == carrier) & (df['ORIGIN_AIRPORT'] == id_airport)]
        # _______________________________________
        # remove extreme values before fitting
        if extrem_values:
            df2['DEPARTURE_DELAY'] = df2['DEPARTURE_DELAY'].apply(lambda x: x if x < 60 else np.nan)
            df2.dropna(how='any')
        # __________________________________

        df2.sort_values('SCHEDULED_DEPARTURE', inplace=True)
        df2['schedule_depart'] = df2['SCHEDULED_DEPARTURE'].apply(lambda x: x.time())
        # ___________________________________________________________________

        test2 = df2['DEPARTURE_DELAY'].groupby(df2['schedule_depart']).apply(get_stats).unstack()
        test2.reset_index(inplace=True)
        # ___________________________________

        fct = lambda x: x.hour * 60 + x.minute
        test2.reset_index(inplace=True)
        test2['schedule_depart_mnts'] = test2['schedule_depart'].apply(fct)
        return test2

    def create_df(df, carrier, id_airport, extrem_values=False):
        df2 = df[(df['AIRLINE'] == carrier) & (df['ORIGIN_AIRPORT'] == id_airport)]
        df2.dropna(how='any', inplace=True)
        df2['weekday'] = df2['SCHEDULED_DEPARTURE'].apply(lambda x: x.weekday())
        # ____________________
        # delete delays > 1h
        df2['DEPARTURE_DELAY'] = df2['DEPARTURE_DELAY'].apply(lambda x: x if x < 60 else np.nan)
        df2.dropna(how='any', inplace=True)
        # _________________
        # formating times
        fct = lambda x: x.hour * 60 + x.minute
        df2['schedule_depart'] = df2['SCHEDULED_DEPARTURE'].apply(lambda x: x.time())
        df2['schedule_depart_mnts'] = df2['schedule_depart'].apply(fct)
        df2['schedule_arrivee'] = df2['SCHEDULED_ARRIVAL'].apply(fct)
        df3 = df2.groupby(['schedule_depart_mnts', 'schedule_arrivee'],
                          as_index=False).mean()
        return df3

    #
    # In[39]:
    # Linear Regression
    ####### Linear_Train #######

    test2 = get_flight_delays(df, carrier, id_airport, False)
    test2.to_csv('Model_dataset.csv', sep=',')

    test = test2[['mean', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_L_train = np.array(test['schedule_depart_mnts'])
    Y_L_train = np.array(test['mean'])
    X_L_train = X_L_train.reshape(len(X_L_train), 1)
    Y_L_train = Y_L_train.reshape(len(Y_L_train), 1)
    regr = linear_model.LinearRegression()
    regr.fit(X_L_train, Y_L_train)
    result_L_train = regr.predict(X_L_train)
    score_L_train = regr.score(X_L_train, Y_L_train)

    # print("R^2 for Linear Train= ",score_L_train)
    print("MSE Linear Train=", metrics.mean_squared_error(result_L_train, Y_L_train))

    # The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
    # ((y_true - y_pred) ** 2).sum() and v is the
    # total sum of squares ((y_true - y_true.mean()) ** 2).sum().

    ####### Linear_Test #######
    test2 = get_flight_delays(df_test, carrier, id_airport, False)

    test = test2[['mean', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_L_test = np.array(test['schedule_depart_mnts'])
    Y_L_test = np.array(test['mean'])
    X_L_test = X_L_test.reshape(len(X_L_test), 1)
    Y_L_test = Y_L_test.reshape(len(Y_L_test), 1)
    result_L_test = regr.predict(X_L_test)
    score_L_test = regr.score(X_L_test, Y_L_test)

    # print("R^2 for Linear Test= ",score_L_test)
    print("MSE Linear Test=", metrics.mean_squared_error(result_L_test, Y_L_test))
    fig1 = Figure_style(8, 4, 1, 1)
    fig1.pos_update(0, 0)
    # fig1.cust_scatter(df1['heure_depart'], df1['DEPARTURE_DELAY'], markeredge = True)
    fig1.cust_plot(X_L_test, Y_L_test, color='b', linestyle=':', linewidth=2, marker=('b', 's', 10))
    fig1.cust_plot(X_L_test, result_L_test, color='g', linewidth=3)
    fig1.style()
    fig1.set_ylabel('Delay (minutes)', fontsize=14)
    fig1.set_xlabel('Departure time', fontsize=14)
    # ____________________________________
    # convert and set the x ticks labels
    fct_convert = lambda x: (int(x / 3600), int(divmod(x, 3600)[1] / 60))
    fig1.axs[fig1.ix, fig1.iy].set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*fct_convert(x))
                                                for x in fig1.axs[fig1.ix, fig1.iy].get_xticks()]);

    # In[77]:
    # Ridge Regression
    ####### Ridge_Training #######
    df3 = get_flight_delays(df, carrier, id_airport)
    df3[:5]
    # df1 = df[(df['AIRLINE'] == carrier) & (df['ORIGIN_AIRPORT'] == id_airport)]
    # df1['heure_depart'] =  df1['SCHEDULED_DEPARTURE'].apply(lambda x:x.time())
    # df1['heure_depart'] = df1['heure_depart'].apply(lambda x:x.hour*60+x.minute)
    df3 = df3[['mean', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X = np.array(df3['schedule_depart_mnts'])
    Y = np.array(df3['mean'])
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)

    parameters = [0.2, 1]
    ridgereg = Ridge(alpha=parameters[0], normalize=True)
    poly = PolynomialFeatures(degree=parameters[1])
    X_ = poly.fit_transform(X)
    ridgereg.fit(X_, Y)
    result_R_train = ridgereg.predict(X_)
    score_R_train = metrics.mean_squared_error(result_R_train, Y)
    r2_R_train = regr.score(X, Y)
    # print("R^2 for Ridge Train:",r2_R_train )
    print('MSE Ridge Train= {}'.format(round(score_R_train, 2)))


    ####### Ridge_Test #######

    df3 = get_flight_delays(df_test, carrier, id_airport)
    df3[:5]

    test = df3[['mean', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_L_test = np.array(test['schedule_depart_mnts'])
    Y_L_test = np.array(test['mean'])
    X_testt = X.reshape(len(X), 1)
    Y_testt = Y.reshape(len(Y), 1)

    X_ = poly.fit_transform(X_testt)
    result_test = ridgereg.predict(X_)

    score_R_test = metrics.mean_squared_error(result_test, Y_testt)

    r2_ridge_test = r2_score(X_testt, Y_testt)
    # print("R^2 for Ridge Test is: ",r2_ridge_test )
    print('MSE Ridge Test = {}'.format(round(np.sqrt(score_R_test), 2)))
    # 'Ecart = {:.2f} min'.format(np.sqrt(score_R_test))

    fig1 = Figure_style(8, 4, 1, 1)
    fig1.pos_update(0, 0)
    # fig1.cust_scatter(df1['heure_depart'], df1['DEPARTURE_DELAY'], markeredge = True)
    fig1.cust_plot(X_testt, Y_testt, color='b', linestyle=':', linewidth=2, marker=('b', 's', 10))
    fig1.cust_plot(X_testt, result_test, color='g', linewidth=3)
    fig1.style()
    fig1.set_ylabel('Delay (minutes)', fontsize=14)
    fig1.set_xlabel('Departure time', fontsize=14)
    # ____________________________________
    # convert and set the x ticks labels
    fct_convert = lambda x: (int(x / 3600), int(divmod(x, 3600)[1] / 60))
    fig1.axs[fig1.ix, fig1.iy].set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*fct_convert(x))
                                                for x in fig1.axs[fig1.ix, fig1.iy].get_xticks()]);

    # %%
    ###########################################################################
    ####### Random Forest_Train #######
    df4 = create_df(df, carrier, id_airport)
    # X_rf_Train = np.array(df3[['schedule_depart','schedule_arrivee', 'ARRIVAL_DELAY', 'SCHEDULED_TIME','ELAPSED_TIME','weekday']])
    # X_rf_Train = np.hstack((X_rf_Train))
    df4 = df4[['DEPARTURE_DELAY', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_rf_Train = np.array(df4['schedule_depart_mnts'])
    Y_rf_Train = np.array(df4['DEPARTURE_DELAY'])

    X_rf_Train = X_rf_Train.reshape(len(X_rf_Train), 1)
    Y_rf_Train = Y_rf_Train.reshape(len(Y_rf_Train), 1)

    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=123456)
    rf.fit(X_rf_Train, Y_rf_Train)

    predicted_train = rf.predict(X_rf_Train)

    test_score = r2_score(Y_rf_Train, predicted_train)
    spearman = spearmanr(Y_rf_Train, predicted_train)
    # pearson = pearsonr(Y_rf_Train, predicted_train)

    # print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    # print(f'Test data R-2 score: {test_score:>5.3}')
    # print(f'Test data Spearman correlation: {spearman[0]:.3}')

    # print("R^2 for RF Train:",test_score )
    print('MSE RF Train= {}'.format(round(metrics.mean_squared_error(predicted_train, Y_rf_Train), 2)))
    # print(f'Test data Pearson correlation: {pearson[0]:.3}')

    ####### Random Forest_Test #######
    df41 = create_df(df_test, carrier, id_airport)
    # X_rf_Train = np.array(df3[['schedule_depart','schedule_arrivee', 'ARRIVAL_DELAY', 'SCHEDULED_TIME','ELAPSED_TIME','weekday']])
    # X_rf_Train = np.hstack((X_rf_Train))
    df41 = df41[['DEPARTURE_DELAY', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_rf_Test = np.array(df41['schedule_depart_mnts'])
    Y_rf_Test = np.array(df41['DEPARTURE_DELAY'])

    X_rf_Test = X_rf_Test.reshape(len(X_rf_Test), 1)
    Y_rf_Test = Y_rf_Test.reshape(len(Y_rf_Test), 1)

    predicted_test = rf.predict(X_rf_Test)

    test_score = r2_score(Y_rf_Test, predicted_test)
    spearman = spearmanr(Y_rf_Test, predicted_test)
    # pearson = pearsonr(Y_rf_Train, predicted_train)

    # print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    # print(f'Test data R-2 score: {test_score:>5.3}')
    # print(f'Test data Spearman correlation: {spearman[0]:.3}')

    score_rf_test = r2_score(X_rf_Test, Y_rf_Test)
    # print("R^2 for RF Test: ",score_rf_test )
    score_RF_test = metrics.mean_squared_error(predicted_test, Y_rf_Test)
    print(' MSE RF Test = {}'.format(round(score_RF_test, 2)))

    fig1 = Figure_style(8, 4, 1, 1)
    fig1.pos_update(0, 0)
    # fig1.cust_scatter(df1['heure_depart'], df1['DEPARTURE_DELAY'], markeredge = True)
    fig1.cust_plot(X_rf_Test, Y_rf_Test, color='b', linestyle=':', linewidth=2, marker=('b', 's', 10))
    fig1.cust_plot(X_rf_Test, predicted_test, color='g', linewidth=3)
    fig1.style()
    fig1.set_ylabel('Delay (minutes)', fontsize=14)
    fig1.set_xlabel('Departure time', fontsize=14)
    # ____________________________________
    # convert and set the x ticks labels
    fct_convert = lambda x: (int(x / 3600), int(divmod(x, 3600)[1] / 60))
    fig1.axs[fig1.ix, fig1.iy].set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*fct_convert(x))
                                                for x in fig1.axs[fig1.ix, fig1.iy].get_xticks()]);

    # %%
    ###########################################################################
    ####### Neural Network_Train #######

    df5 = create_df(df, carrier, id_airport)
    # X_rf_Train = np.array(df3[['schedule_depart','schedule_arrivee', 'ARRIVAL_DELAY', 'SCHEDULED_TIME','ELAPSED_TIME','weekday']])
    # X_rf_Train = np.hstack((X_rf_Train))
    df5 = df5[['DEPARTURE_DELAY', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_nn_Train = np.array(df5['schedule_depart_mnts'])
    Y_nn_Train = np.array(df5['DEPARTURE_DELAY'])

    X_nn_Train = X_nn_Train.reshape(len(X_nn_Train), 1)
    Y_nn_Train = Y_nn_Train.reshape(len(Y_nn_Train), 1)

    regr = LinearSVR(random_state=0)
    #    from sknn.mlp import Classifier, Layer
    #    #regr = LinearSVR(random_state=0)
    #    regr = Classifier(
    #    layers=[
    #        Layer("Rectifier", units=10),
    #        Layer("Linear")],
    #    learning_rate=0.02,
    #    n_iter=5)
    regr.fit(X_nn_Train, Y_nn_Train)

    predict_train_NN = regr.predict(X_nn_Train)

    r2_NN_train = r2_score(Y_nn_Train, predict_train_NN)
    # print("R^2 for NN Train:",r2_NN_train )
    print('MSE NN Train= {}'.format(round(metrics.mean_squared_error(predict_train_NN, Y_nn_Train), 2)))

    ####### Neural Network_Test #######
    df51 = create_df(df_test, carrier, id_airport)
    # X_rf_Train = np.array(df3[['schedule_depart','schedule_arrivee', 'ARRIVAL_DELAY', 'SCHEDULED_TIME','ELAPSED_TIME','weekday']])
    # X_rf_Train = np.hstack((X_rf_Train))
    df51 = df51[['DEPARTURE_DELAY', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_NN_Test = np.array(df51['schedule_depart_mnts'])
    Y_NN_Test = np.array(df51['DEPARTURE_DELAY'])

    X_NN_Test = X_NN_Test.reshape(len(X_NN_Test), 1)
    Y_NN_Test = Y_NN_Test.reshape(len(Y_NN_Test), 1)

    predict_test_NN = regr.predict(X_NN_Test)

    score_NN_test = r2_score(X_NN_Test, Y_NN_Test)
    # print("R^2 for NN Test: ",score_NN_test )
    MSE_NN_test = metrics.mean_squared_error(predict_test_NN, Y_NN_Test)
    print('MSE NN Test = {}'.format(round(MSE_NN_test, 2)))

    fig1 = Figure_style(8, 4, 1, 1)
    fig1.pos_update(0, 0)
    # fig1.cust_scatter(df1['heure_depart'], df1['DEPARTURE_DELAY'], markeredge = True)
    fig1.cust_plot(X_NN_Test, Y_NN_Test, color='b', linestyle=':', linewidth=2, marker=('b', 's', 10))
    fig1.cust_plot(X_NN_Test, predict_test_NN, color='g', linewidth=3)
    fig1.style()
    fig1.set_ylabel('Delay (minutes)', fontsize=14)
    fig1.set_xlabel('Departure time', fontsize=14)

    # convert and set the x ticks labels
    fct_convert = lambda x: (int(x / 3600), int(divmod(x, 3600)[1] / 60))
    fig1.axs[fig1.ix, fig1.iy].set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*fct_convert(x))
                                                for x in fig1.axs[fig1.ix, fig1.iy].get_xticks()]);

    # %%

    ###########################################################################
    ####### SVM_Train #######

    df6 = create_df(df, carrier, id_airport)
    df6 = df6[['DEPARTURE_DELAY', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_svm_Train = np.array(df6['schedule_depart_mnts'])
    Y_svm_Train = np.array(df6['DEPARTURE_DELAY'])

    X_svm_Train = X_svm_Train.reshape(len(X_svm_Train), 1)
    Y_svm_Train = Y_svm_Train.reshape(len(Y_svm_Train), 1)

    regr = SVR(kernel='linear')

    regr.fit(X_svm_Train, Y_svm_Train)

    predict_train_svm = regr.predict(X_svm_Train)
    r2_svm_train = r2_score(Y_nn_Train, predict_train_svm)
    # print("R^2 for svm Train:",r2_svm_train )
    print('MSE svm Train= {}'.format(round(metrics.mean_squared_error(predict_train_svm, Y_svm_Train), 2)))

    ####### SVM_Test #######
    df61 = create_df(df_test, carrier, id_airport)
    # X_rf_Train = np.array(df3[['schedule_depart','schedule_arrivee', 'ARRIVAL_DELAY', 'SCHEDULED_TIME','ELAPSED_TIME','weekday']])
    # X_rf_Train = np.hstack((X_rf_Train))
    df61 = df61[['DEPARTURE_DELAY', 'schedule_depart_mnts']].dropna(how='any', axis=0)
    X_svm_Test = np.array(df61['schedule_depart_mnts'])
    Y_svm_Test = np.array(df61['DEPARTURE_DELAY'])

    X_svm_Test = X_svm_Test.reshape(len(X_svm_Test), 1)
    Y_svm_Test = Y_svm_Test.reshape(len(Y_svm_Test), 1)

    predict_test_svm = regr.predict(X_svm_Test)

    r2_svm_test = r2_score(X_svm_Test, Y_svm_Test)
    # print("R^2 for svm Test: ",r2_svm_test )
    mse_svm_test = metrics.mean_squared_error(predict_test_svm, Y_svm_Test)
    print('MSE svm Test= {}'.format(round(mse_svm_test, 2)))

    fig1 = Figure_style(8, 4, 1, 1)
    fig1.pos_update(0, 0)
    # fig1.cust_scatter(df1['heure_depart'], df1['DEPARTURE_DELAY'], markeredge = True)
    fig1.cust_plot(X_svm_Test, Y_svm_Test, color='b', linestyle=':', linewidth=2, marker=('b', 's', 10))
    fig1.cust_plot(X_svm_Test, predict_test_svm, color='g', linewidth=3)
    fig1.style()
    fig1.set_ylabel('Delay (minutes)', fontsize=14)
    fig1.set_xlabel('Departure time', fontsize=14)
    # ____________________________________
    # convert and set the x ticks labels
    fct_convert = lambda x: (int(x / 3600), int(divmod(x, 3600)[1] / 60))
    fig1.axs[fig1.ix, fig1.iy].set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*fct_convert(x))
                                                for x in fig1.axs[fig1.ix, fig1.iy].get_xticks()]);

    return np.mean(result_L_test), np.mean(result_test), np.mean(predicted_test), np.mean(predict_test_NN), np.mean(
        predict_test_svm)

@app.route('/default_get_inpt', methods = ['GET', 'POST'])
def default_get_inpt():
    if not os.path.isdir(target):
        os.mkdir(target)
    if request.method == 'POST':
        if request.form.get('org_airport', None) is not None:
            org_airport = request.form.get('org_airport', None)
            airline = request.form.get('airline', None)
            l, r, rf, nn, svm =default_datasets(airline, org_airport)
            return render_template('output.html', l=l, r=r, rf=rf, nn=nn, svm=svm)

        else:
            for file in request.files.getlist("file"):
                fname = file.filename
                dest = "/".join([target, fname])
                file.save(dest)

    ####call the function to read the file data with [dest] and sample the train sample [trn_smpl]


@app.route('/training_input', methods = ['GET', 'POST'])
def training_input():
    if not os.path.isdir(target):
        os.mkdir(target)
    if request.method == 'POST':
        if request.form.get('org_airport', None) is not None:
            org_airport = request.form.get('org_airport', None)
            airline = request.form.get('airline', None)
        else:
            for file in request.files.getlist("file"):
                fname = file.filename
                dest = "/".join([target, fname])
                file.save(dest)

    ####call the function to read the file data with [dest] and sample the train sample [trn_smpl]
    return render_template('output.html')

@app.route('/default_input', methods=['GET', 'POST'])
def default_input():
    return render_template('default_input.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'GET' and request.args['trn_smpl'] is not None:
        trn_smpl = request.args['trn_smpl']

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        fname = file.filename
        dest = "/".join([target, fname])
        print(dest)
        file.save(dest)


    ####call the function to read the file data with [dest] and sample the train sample [trn_smpl]
    return render_template('training_input.html')

    #if request.method == 'POST':
    #    f = request.files['file']
    #    f.save(secure_filename(f.filename))
    #    return 'file uploaded successfully'

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home1.html')

if __name__ == '__main__':
   app.run(debug = True)