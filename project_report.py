#Import Required packages
import mysql.connector
from sqlalchemy import *
import pandas as pd
import numpy as np
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import streamlit as st
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import datetime as dt
import time
start = time.time()
import shutil
import calendar
import os
import sys
import string

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score,r2_score,mean_absolute_percentage_error
from scipy.stats import chi2_contingency

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot

import warnings
warnings.filterwarnings("ignore") 

#Functions
#Establish Connection with SQL Database
def connectDB(host,username,password):
    
    mydb = mysql.connector.connect(
        host = host,
        username = username,
        password = password
    )  
    return mydb

#Convert time to required format
def timeConversion(s):
    if "PM" in s:
        s=s.replace("PM"," ")
        t= s.split(":")
        if t[0] != '12':
            t[0]=str(int(t[0])+12)
            s= (":").join(t)
        return s
    else:
        s = s.replace("AM"," ")
        t= s.split(":")
        if t[0] == '12':
            t[0]='00'
            s= (":").join(t)
        return s

#Extract numerical and categorical columns in the dataset
def columnType(df):
    categorical_cols = [cname for cname in df.columns if
                        df[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in df.columns if 
                    df[cname].dtype in ['int64', 'float64']]

    return categorical_cols,numerical_cols

#Maximum and minimum value for each column
def daily_maximum_values(df,cols):
    col_list=[]
    col_max_list=[]
    col_min_list=[]
    date_max_list=[]
    date_min_list=[]
    for col in cols:
        col_list.append(col)
        col_max = df[col].max()
        col_max_list.append(col_max)
        col_min = df[col].min()
        col_min_list.append(col_min)
        date_max = df[df[col] ==  col_max][["date"]]
        date_max_list.append(date_max)
        date_min = df[df[col] ==  col_min][["date"]]
        date_min_list.append(date_min)

    return col_list,col_max_list,col_min_list,date_max_list,date_min_list


# Time Series Analysis of every continuous column in the dataset.
# Function to calculate and plot mean value of columns grouped by year,month,day,week,etc.
def time_period_mean(df,period,cols):

    mean_time_series_col_list=[]
    mean_time_series_col_dict = {}
    col_list=[]
    col_max_list=[]
    col_min_list=[]
    date_max_list=[]
    date_min_list=[]
    for col in cols:
        for n in range(366):
            if period == "hour":
                df_time = df[df["date"].dt.hour == n][col].mean()
                
            elif period == "month":
                df_time = df[df["date"].dt.month == n][col].mean()

            elif period == "year":
                df_time = df[df["date"].dt.year == (int(df["date"][0].year)+n)][col].mean()

            elif period == "dayofmonth":
                df_time = df[df["date"].dt.day == n][col].mean()

            elif period == "dayofweek":
                df_time = df[df["date"].dt.dayofweek == n][col].mean()

            elif period == "dayofyear":
                df_time = df[df["date"].dt.dayofyear == n][col].mean()

            elif period == "weekofyear":
                df_time = df[df["date"].dt.weekofyear == n][col].mean()
            
            elif period == "quarter":
                df_time = df[df["date"].dt.quarter == n][col].mean()

            mean_time_series_col_list.append([col,n,df_time])
             
        col_list.append(col)
        hourly_mean_df = pd.DataFrame(mean_time_series_col_list,columns = ["Column",period,'Value'])
        max_value = hourly_mean_df[hourly_mean_df["Column"] == col]["Value"].max()
        col_max_list.append(max_value)
        min_value = hourly_mean_df[hourly_mean_df["Column"] == col]["Value"].min()
        col_min_list.append(min_value)
        max_date = hourly_mean_df[[hourly_mean_df["Column"] == col] and hourly_mean_df["Value"] == max_value][period]
        date_max_list.append(max_date)
        min_date = hourly_mean_df[[hourly_mean_df["Column"] == col] and hourly_mean_df["Value"] == min_value][period]
        date_min_list.append(min_date)

    return col_list,col_max_list,col_min_list,date_max_list,date_min_list,hourly_mean_df

#Find top n correlated columns in the dataset
def top_n_correlated_columns(df,n):
    df_corr = abs(df.corr(method = "pearson"))
    np.fill_diagonal(df_corr.values,0)
    # plt.figure(figsize=(16,8))
    # sns.heatmap(df_corr,lw=2,annot=True)
    # plt.show()

    corr_dict={}
    for col in df_corr.columns:
        col_sum = df_corr[col].sum()
        corr_dict.update({col:col_sum})

    sorted_corr_dict = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1],reverse=True)}
    sorted_corr_df = pd.DataFrame(list(sorted_corr_dict.items()),columns=["Feature","Correlation"])

    #Extract and plot top correlated columns
    most_corr_columns = sorted_corr_df["Feature"].values
    
    return sorted_corr_df[:n],most_corr_columns[:n]

def output_df_to_pdf(pdf, df):
    table_cell_width = 20
    table_cell_height = 8
    pdf.set_font('Arial', 'B', 8)

    # Loop over to print column names
    cols = df.columns
    for col in cols:
        pdf.cell(table_cell_width, table_cell_height, col, align='C', border=1)
    # Line break
    pdf.ln(table_cell_height)
    pdf.set_font('Arial', 'B', 8)
    # Loop over to print each data in the table
    for row in df.itertuples():
        for col in cols:
            value = str(getattr(row, col))
            pdf.cell(table_cell_width, table_cell_height,
                     value, align='C', border=1)
        pdf.ln(table_cell_height)


#----------------------------------------------------------Functions for Prophet Model-------------------------------------------------------------------#
# def paramaterTuning(df,col):

#     hp_test = pd.concat([df["date"],df[col]],axis=1)
#     hp_test.rename(columns={"date":"ds",col:"y"},inplace=True)
#     hp_test_train = hp_test.iloc[:int(hp_test.shape[0]*0.8)]
#     hp_test_test = hp_test.iloc[int(hp_test.shape[0]*0.8):]

#     params_grid = {'changepoint_prior_scale':[0.1,0.2],
#                     'n_changepoints' : [50,200]}
#     grid = ParameterGrid(params_grid)
#     cnt = 0
#     for p in grid:
#         cnt = cnt+1

#     print('Total Possible Models',cnt)

#     model_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
#     for p in grid:
#         test = pd.DataFrame()
#         print(p)
#         train_model =Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
#                             n_changepoints = p['n_changepoints'],
#                             seasonality_mode = "multiplicative",
#                             weekly_seasonality=True,
#                             daily_seasonality = True,
#                             yearly_seasonality = True,
#                             interval_width=0.95)

#         train_model.fit(hp_test_train)
#         train_forecast = train_model.make_future_dataframe(periods=int(hp_test.shape[0]*0.2), freq='D',include_history = False)
#         train_forecast = train_model.predict(train_forecast)
#         test=train_forecast[['ds','yhat']]
#         MAPE = mean_absolute_percentage_error(hp_test_test["y"],abs(test['yhat']))
#         print('Mean Absolute Percentage Error(MAPE)------------------------------------',MAPE)
#         model_parameters = model_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)

#         parameters = model_parameters.sort_values(by=['MAPE'])
#         parameters = parameters.reset_index(drop=True)
#         display(parameters.head())

#         tuned_params = list(parameters['Parameters'][0].values())
#         return tuned_params

#Facebook Prophet model to forecast data features.
def prophet_model(df,col):

    # prophet_df = pd.DataFrame()

    # params_list = paramaterTuning(df,col)
    # changepoint_prior_scale = params_list[0]
    # n_changepoints = params_list[1]
    # seasonality_mode = params_list[2]

    model_prophet = Prophet(seasonality_mode="multiplicative",
                            changepoint_prior_scale=0.5,
                            n_changepoints=200,
                            weekly_seasonality=True,
                            daily_seasonality = True,
                            yearly_seasonality = True,
                            interval_width=0.85)
    #Prophet preprocessing
    prophet_df = pd.concat([df.iloc[:,0],df[col].rolling(50).mean()],axis=1)
    prophet_df.rename(columns={prophet_df.columns[0] : "ds",col:'y'},inplace=True)
    X = prophet_df[["ds",'y']]

    model_prophet.fit(X)
    n = prophet_df.shape[0]
    prophet_future = model_prophet.make_future_dataframe(periods = int(0.1*n))
    prophet_forecast = model_prophet.predict(prophet_future)
    # display(prophet_forecast[['ds','yhat','yhat_lower','yhat_upper']])
    prophet_forecast_predictions = prophet_forecast[round(-0.1*n+1):].reset_index()

    model_prophet.plot_components(prophet_forecast)
    plt.show()

    return prophet_df, prophet_forecast, prophet_forecast_predictions


def prophet_plot(prophet_forecast_predictions):
    plt.plot(prophet_forecast_predictions["ds"],prophet_forecast_predictions["yhat"],label="Prediction")
    plt.plot(prophet_forecast_predictions["ds"],prophet_forecast_predictions["yhat_upper"],label="Upper CF")
    plt.plot(prophet_forecast_predictions["ds"],prophet_forecast_predictions["yhat_lower"],label="Lower CF")
    plt.grid()
    plt.legend() 
    plt.show()

#Generate statements about the percentage increase/decrease for important columns.
def forecasted_strings(prophet_df,prophet_forecast_predictions,col,percent,type):
    last_day = prophet_df.iloc[-1]
    # n = prophet_df.shape[0]
    global forecast_string

    for i in range(prophet_forecast_predictions.shape[0]):
        if type == "Jump":
            if (prophet_forecast_predictions["yhat"].iloc[i] >= int(((100+percent)/100)*last_day["y"])):
                # print(prophet_forecast_predictions[["ds","yhat"]].iloc[i].values)
                # print(col," will jump", percent ,"% in ",str(prophet_forecast_predictions["ds"].iloc[i].date() - last_day["ds"])[:-9],"(",prophet_forecast_predictions["ds"].iloc[i].date(),"), with a value of",prophet_forecast_predictions["yhat"].iloc[i])
                forecast_string = str(col) +" will jump " +str(percent) + "% in "+str(prophet_forecast_predictions["ds"].iloc[i].date() - last_day["ds"])[:-9]+" ("+str(prophet_forecast_predictions["ds"].iloc[i].date())+"), with a value of "+str(round(prophet_forecast_predictions["yhat"].iloc[i],3))
                return forecast_string
            
            
        elif type == "Drop":    
            if (prophet_forecast_predictions["yhat"].iloc[i] <= int(((100-percent)/100)*last_day["y"])):
                # print(prophet_forecast_predictions[["ds","yhat"]].iloc[i].values)
                # print(col," will jump",percent ,"% in ",str(prophet_forecast_predictions["ds"].iloc[i].date() - last_day["ds"])[:-9],"(",prophet_forecast_predictions["ds"].iloc[i].date(),"), with a value of",prophet_forecast_predictions["yhat"].iloc[i])
                forecast_string = str(col) +" will drop " +str(percent) + "% in "+str(prophet_forecast_predictions["ds"].iloc[i].date() - last_day["ds"])[:-9]+" ("+str(prophet_forecast_predictions["ds"].iloc[i].date())+"), with a value of "+str(round(prophet_forecast_predictions["yhat"].iloc[i],3))
                return forecast_string
            
        else:
            print("Type should be either 'Jump' or 'Drop'")


#Supervised Data analysis.
def column_feature_importance_target(df,num_columns,target):

    X = df[num_columns].drop(["Temperature"],axis=1)
    Y = df[target]

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

    # model_xgb_test = xgb.XGBRegressor(n_estimators=2000,max_depth=50)
    # model_xgb_test.fit(X_train, Y_train,
    #         eval_set=[(X_train, Y_train), (X_test, Y_test)],
    #         early_stopping_rounds=50,
    #         verbose=False)

    dtrain = xgb.DMatrix(X,label=Y,enable_categorical=True)
    watchlist = [(dtrain, 'train')]
    param = {'max_depth': 150, 'learning_rate': 0.05}
    num_round=200
    model_xgb_test = xgb.train(param, dtrain, num_round, watchlist)

    # pred_test = model_xgb_test.predict(X_test)
    pred_test = model_xgb_test.predict(xgb.DMatrix(X_test))

    print("Mean Absolute Error (MAE): ",round(mean_absolute_error(Y_test,pred_test),6))
    print("Mean Squared Error (MSE): ",round(mean_squared_error(Y_test,pred_test),6))
    print("R2 Score: ",round(r2_score(Y_test,pred_test),5)*100)

    plot_importance(model_xgb_test, height=0.8)
    plt.title("Feature Importance")
    plt.show()

    # plt.scatter(model_xgb_test.feature_importances_,num_columns)
    # plt.plot(model_xgb_test.feature_importances_,num_columns)

    scores = model_xgb_test.get_score(importance_type = "weight")
    cols = pd.DataFrame(scores.values() ,columns=["importance"],index = scores.keys()).sort_values(by="importance",ascending=False).reset_index().rename(columns = {"index":"feature"})
    print(cols)
    return cols


#----------------------------------------------Datetime feature importance------------------------------------------------------#
def create_features(df,label=None):

    df.index = df["date"]
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth']]
    if label:
        y = df[label]
        return X, y
    return X

def column_feature_importance(df,num_columns):

    #Calculate time series feature importance using XGBoost
    model_xgb = xgb.XGBRegressor(n_estimators=1000,max_depth=50)
    col_cols_dict = {}
    for col in num_columns:
        X,Y = create_features(df,label=col)

        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
        model_xgb.fit(X_train, Y_train,
                eval_set=[(X_train, Y_train), (X_test, Y_test)],
                early_stopping_rounds=50,
                verbose=False)

        pred_test = model_xgb.predict(X_test)

        # print("---------------",col,"-----------------")
        # print("Mean Absolute Error (MAE): ",round(mean_absolute_error(Y_test,pred_test),6))
        # print("Mean Squared Error (MSE): ",round(mean_squared_error(Y_test,pred_test),6))
        # print("R2 Score: ",round(r2_score(Y_test,pred_test),5)*100)

        # plot_importance(model_xgb, height=0.8)
        # plt.title(col)
        # plt.show() 

        scores = model_xgb.get_booster().get_score(importance_type='weight')
        cols = pd.DataFrame(scores.values() ,columns=["importance"],index = scores.keys()).sort_values(by="importance",ascending=False).reset_index().rename(columns = {"index":"feature"})
        col_cols_dict.update({col:cols})

    return col_cols_dict

#Create cover page for the report PDF.
def cover_page(FPDF):
    pdf = FPDF()
    pdf.add_page()
    pdf.image("cover_page.png",0,0,210,297,type="PNG")

    return pdf

def load_data_method(data_load_choice):

    if(data_load_choice == 1):
        df = pd.read_csv(data_name)
        print("Data Successfully Added!")
        print("Generating Report.......")
        return df
    elif(data_load_choice == 2):
        df = pd.read_sql(fetch_query,mydb)
        print("Data Successfully Added!")
        print("Generating Report.......")
        return df
    else:
        print("Not a valid input. Type either 1 or 2")


#----------------------------------------------------------------------Start main-----------------------------------------------------------------------------#

#Create PDF object and set properties
generated_pdf = cover_page(FPDF)
generated_pdf.add_page()
generated_pdf.set_fill_color(169,169,169)
generated_pdf.set_font("Arial",'BU',26)
generated_pdf.cell(0,10,"Report Generated",align='C')
generated_pdf.ln(20)

#Establish Connection with DB
mydb = connectDB("localhost","root","test123")
schema_used = "project_test."
data_used = "weather_data_denver"
fetch_query = "SELECT * FROM "+schema_used+data_used+" LIMIT 50000"
#Call data directly using csv file.
data_name = "weather_data_denver.csv"

#Get data loading method.
data_load_choice = int(input("How do you want to load your data? (1 - Directly from CSV / 2 - From SQL Database)"))
df = load_data_method(data_load_choice)


#Data Preprocessing
if("Time" in df.columns):
    for i in range(df.shape[0]):
        if "AM" in df["Time"][i]: 
            df["Time"][i] = timeConversion(df["Time"][i])

    df["Date"] = df["Date"].map(str) + " " +df["Time"]
    df["Date"] = pd.to_datetime(df["Date"],infer_datetime_format=True)
    df.drop(["Time"],inplace=True,axis=1)

df.rename(columns={df.columns[0] : "date"},inplace=true)
df["date"] = pd.to_datetime(df["date"],infer_datetime_format=True)
df.fillna("ffill")

#Remove punctuation from df column names
cols_df=[]
for col in df.columns:
    col = col.translate(str.maketrans('','',string.punctuation))
    cols_df.append(col)
df.columns = cols_df

#Add table used
df_report = df.copy()
df_report["date"] = df_report["date"].dt.strftime('%d/%m/%Y').astype(str)

numeric_cols = df_report.select_dtypes(include='number').columns
df_report[numeric_cols] = df_report[numeric_cols].round(2)
generated_pdf.set_font("Arial",'BI',12)
generated_pdf.cell(30,10,txt="Table Used",align="L",ln=1,border=True,fill=True)
output_df_to_pdf(generated_pdf,df_report.head(10))

#Display description of the dataframe
print(df.head())

cat_columns,num_columns = columnType(df) 

#Treating outliers
v_col_list = []
for k, v in df[num_columns].items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
        if perc > 50:
                v_col_list.append(v_col.name)
        # print("Column {} outliers = {} => {}%".format(k,len(v_col),round((perc),3)))

#Drop columns with more than 50% outliers       
df.drop(v_col_list,inplace=True,axis=1)
cat_columns,num_columns = columnType(df) 

#Group dataset by days.
daily_grouped_df = df.groupby([df.iloc[:,0].dt.date]).mean().reset_index()
daily_grouped_df.head()

#Add summary table
generated_pdf.ln(5)
description_report = df.describe().round(3).reset_index()
print(description_report)
generated_pdf.set_font("Arial",'BI',12)
generated_pdf.cell(45,10,txt="Summary of Table",align="L",ln=1,border=True,fill=True)
output_df_to_pdf(generated_pdf,description_report)


#Monthly and yearly average values
monthly_grouped_df = df.groupby([df["date"].dt.month]).mean().reset_index().rename(columns = {"date":"Month"}).round(3)
monthly_grouped_df["month"] = ["January","February","March","April","May","June","July","August","September","October","November","December"]
# monthly_grouped_df["Month"] = ["January","February","March","April"]
print(monthly_grouped_df)
generated_pdf.ln(5)
generated_pdf.set_font("Arial",'BI',12)
generated_pdf.cell(70,10,txt="Monthly grouped values (Mean)",align="L",ln=1,border=True,fill=True)
output_df_to_pdf(generated_pdf,monthly_grouped_df)

yearly_grouped_df = df.groupby([df["date"].dt.year]).mean().reset_index().rename(columns = {"date":"Year"}).round(3)
print(yearly_grouped_df)
generated_pdf.ln(5)
generated_pdf.set_font("Arial",'BI',12)
generated_pdf.cell(70,10,txt="Yearly grouped values (Mean)",align="L",ln=1,border=True,fill=True)
output_df_to_pdf(generated_pdf,yearly_grouped_df)

n=5
m=3
most_corr_columns_df,most_corr_columns = top_n_correlated_columns(df,n)
print("Top",m,"correlated columns: ")
print(most_corr_columns_df[:m].round(3))
generated_pdf.ln(5)
generated_pdf.set_font("Arial",'BI',12)
generated_pdf.cell(32,10,txt = "Top "+str(m)+ " features ",align="L",ln=1,border=True,fill=True)
output_df_to_pdf(generated_pdf,most_corr_columns_df[:m].round(3))

# Datetime Feature Importance
df_xgb = df.copy()
dt_n=3
col_cols_dict = column_feature_importance(df_xgb,most_corr_columns)
reformed_dict = {}
for outerKey, innerDict in col_cols_dict.items():
    for innerKey, values in innerDict.items():
        reformed_dict[(outerKey,
                       innerKey)] = values

col_cols_df = pd.DataFrame(reformed_dict)
print(col_cols_df[:dt_n])


#-------------------------------------------------------------------------Output to PDF-----------------------------------------------------------------------#

#1 - Highest and lowest values
col_list,col_max_list,col_min_list,date_max_list,date_min_list = daily_maximum_values(df,most_corr_columns[:n])
generated_pdf.ln(30)
generated_pdf.set_font("Arial",'BI',16)
generated_pdf.cell(0,10,txt="Highest and lowest values in " +str(n)+ " important columns ",ln=1,align="C",border=1,fill=True)
generated_pdf.ln(10)
generated_pdf.set_font("Arial",'B',12)
for i in range(len(col_list)):
    print("The maximum ",col_list[i]," was ",round(col_max_list[i],2),"on ",pd.Timestamp(date_max_list[i].values[0][0]).strftime("%d-%m-%Y"),"at ",pd.Timestamp(date_max_list[i].values[0][0]).strftime("%H-%M"))
    print("The minimum ",col_list[i]," was ",round(col_min_list[i],2),"on ",pd.Timestamp(date_min_list[i].values[0][0]).strftime("%d-%m-%Y"),"at ",pd.Timestamp(date_max_list[i].values[0][0]).strftime("%H-%M"),"\n")
    generated_pdf.cell(0,10,txt ="The maximum "+str(col_list[i])+" was "+str(round(col_max_list[i],2))+" on "+str(pd.Timestamp(date_max_list[i].values[0][0]).strftime("%d-%m-%Y")+" at "+str(pd.Timestamp(date_max_list[i].values[0][0]).strftime("%H:%M"))),align="L",ln=1)
    generated_pdf.cell(0,10,txt ="The minimum "+str(col_list[i])+" was "+str(round(col_min_list[i],2))+" on "+str(pd.Timestamp(date_min_list[i].values[0][0]).strftime("%d-%m-%Y")+" at "+str(pd.Timestamp(date_min_list[i].values[0][0]).strftime("%H:%M"))),align="L",ln=1)
    generated_pdf.cell(0,10,txt ="--------------------------------------------------------------------------------------------------------------",align="L",ln=1)

#2 - Time Period Analysis
generated_pdf.ln(20)
generated_pdf.set_font("Arial",'BI',16)
generated_pdf.cell(0,10,txt="Time period analysis ",ln=1,align="C",border=True,fill=True)
generated_pdf.ln(10)
generated_pdf.set_font("Arial",'B',12)
c=0
for colm in most_corr_columns:
    generated_pdf.cell(0,10,txt=colm,ln=1,align="C",border=True)
    for per in col_cols_df[:dt_n][colm]["feature"].values:
        col_list,col_max_list,col_min_list,date_max_list,date_min_list,hourly_mean_df = time_period_mean(df,per,most_corr_columns)
        # for i in range(len(col_list)):
        if per == "hour":
            print(col_list[c],"is highest between",date_max_list[c].values[0],":00 and",date_max_list[c].values[0]+1,":00 with an average value of",round(col_max_list[c],3))
            print(col_list[c],"is lowest between",date_min_list[c].values[0],":00 and",date_min_list[c].values[0]+1,":00 with an average value of",round(col_min_list[c],3))
            generated_pdf.set_fill_color(230,230,230)
            generated_pdf.cell(0,5,fill=True,txt =str(col_list[c])+" is highest between "+str(date_max_list[c].values[0])+":00 and "+str(date_max_list[c].values[0]+1)+":00 with an average value of "+str(round(col_max_list[c],3)),align="L",ln=1)
            generated_pdf.cell(0,8,fill=True,txt =str(col_list[c])+" is lowest between "+str(date_min_list[c].values[0])+":00 and "+str(date_min_list[c].values[0]+1)+":00 with an average value of "+str(round(col_min_list[c],3)),align="L",ln=1)
                    
        elif per == "month":
            quarters = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
            k,v = quarters.keys() , quarters.values()
            print(date_max_list)
            res = [key for key in quarters if quarters[key] == date_max_list[c].values[0]]
            print(col_list[c],"is highest in",res[0],"with an average value of",round(col_max_list[c],3))
            generated_pdf.set_fill_color(225,225,225)
            generated_pdf.cell(0,5,fill=True,txt = str(col_list[c])+"is highest in"+str(res[0])+"with an average value of"+str(round(col_max_list[c],3)),align="L",ln=1)
            res = [key for key in quarters if quarters[key] == date_min_list[c].values[0]]
            print(col_list[c],"is lowest in",res[0],"with an average value of",round(col_min_list[c],3))
            generated_pdf.cell(0,8,fill=True,txt = str(col_list[c])+"is lowest in"+str(res[0])+"with an average value of"+str(round(col_min_list[c],3)),align="L",ln=1)

        elif per == "year":
            print(col_list[c],"is highest in year",date_max_list[c].values[0],"with an average value of",round(col_max_list[c],3))
            print(col_list[c],"is lowest in year",date_min_list[c].values[0],"with an average value of",round(col_min_list[c],3))
            generated_pdf.set_fill_color(220, 220, 220)
            generated_pdf.cell(0,5,fill=True,txt =str(col_list[c])+" is highest in year "+str(date_max_list[c].values[0])+" with an average value of "+str(round(col_max_list[c],3)),align="L",ln=1)
            generated_pdf.cell(0,8,fill=True,txt =str(col_list[c])+" is lowest in year "+str(date_min_list[c].values[0])+" with an average value of "+str(round(col_min_list[c],3)),align="L",ln=1)

        elif per == "dayofmonth":
            print(col_list[c],"is highest between day",date_max_list[c].values[0],"and day",date_max_list[c].values[0]+1," with an average value of ",round(col_max_list[c],3))
            print(col_list[c],"is lowest between day",date_min_list[c].values[0],"and day",date_min_list[c].values[0]+1," with an average value of ",round(col_min_list[c],3))
            generated_pdf.set_fill_color(215, 215, 215)
            generated_pdf.cell(0,5,fill=True,txt =str(col_list[c])+" is highest on day "+str(date_max_list[c].values[0])+" with an average value of "+str(round(col_max_list[c],3)),align="L",ln=1)
            generated_pdf.cell(0,8,fill=True,txt =str(col_list[c])+" is lowest on day "+str(date_min_list[c].values[0])+" with an average value of "+str(round(col_min_list[c],3)),align="L",ln=1)

        elif per == "dayofweek":
            quarters = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            k,v = quarters.keys() , quarters.values()
            res = [key for key in quarters if quarters[key] == date_max_list[c].values[0]]
            print(col_list[c],"is highest on",res[0],"with an average value of",round(col_max_list[c],3))
            generated_pdf.set_fill_color(210, 210, 210)
            generated_pdf.cell(0,5,fill=True,txt =str(col_list[c])+" is highest on "+str(res[0])+" with an average value of "+str(round(col_max_list[c],3)),align="L",ln=1)
            res = [key for key in quarters if quarters[key] == date_min_list[c].values[0]]
            print(col_list[c],"is lowest on",res[0],"with an average value of",round(col_min_list[c],3))
            generated_pdf.cell(0,8,fill=True,txt =str(col_list[c])+" is lowest on "+str(res[0])+" with an average value of "+str(round(col_min_list[c],3)),align="L",ln=1)

        elif per == "dayofyear":
            print(col_list[c],"is highest on the",date_max_list[c].values[0],"day of the year with an average value of",round(col_max_list[c],3))
            print(col_list[c],"is lowest on the",date_min_list[c].values[0],"day of the year with an average value of",round(col_min_list[c],3))
            generated_pdf.set_fill_color(205, 205, 205)
            generated_pdf.cell(0,5,fill=True,txt =str(col_list[c])+" is highest on the "+str(date_max_list[c].values[0])+" day of the year with an average value of "+str(round(col_max_list[c],3)),align="L",ln=1)
            generated_pdf.cell(0,8,fill=True,txt =str(col_list[c])+" is lowest on the "+str(date_min_list[c].values[0])+" day of the year with an average value of "+str(round(col_min_list[c],3)),align="L",ln=1)

        elif per == "weekofyear":
            print(col_list[c],"is highest in week number",date_max_list[c].values[0],"with an average value of",round(col_max_list[c],3))
            print(col_list[c],"is highest in week number",date_min_list[c].values[0],"with an average value of",round(col_min_list[c],3))
            generated_pdf.set_fill_color(200,200,200)
            generated_pdf.cell(0,5,fill=True,txt =str(col_list[c])+" is highest in week number "+str(date_max_list[c].values[0])+" with an average value of "+str(round(col_max_list[c],3)),align="L",ln=1)
            generated_pdf.cell(0,8,fill=True,txt =str(col_list[c])+" is lowest in week number "+str(date_min_list[c].values[0])+" with an average value of "+str(round(col_min_list[c],3)),align="L",ln=1)

        elif per == "quarter":
            quarters = {"1st":1,"2nd":2,"3rd":3,"4th":4}
            k,v = quarters.keys() , quarters.values()
            res = [key for key in quarters if quarters[key] == date_max_list[c].values[0]]
            print(col_list[c],"is highest in the",res[0],"quarter with an average value of",round(col_max_list[c],3))
            generated_pdf.set_fill_color(190,190,190)
            generated_pdf.cell(0,5,fill=True,txt =str(col_list[c])+" is highest in the "+str(res[0])+" quarter with an average value of "+str(round(col_max_list[c],3)),align="L",ln=1)
            res = [key for key in quarters if quarters[key] == date_min_list[c].values[0]]
            print(col_list[c],"is lowest in the",res[0],"quarter with an average value of",round(col_min_list[c],3))
            generated_pdf.cell(0,8,fill=True,txt =str(col_list[c])+" is highest in the "+str(res[0])+" quarter with an average value of "+str(round(col_min_list[c],3)),align="L",ln=1)

    c = c+1
    generated_pdf.ln(10)
        # # Plot these trends
        # plt.figure(figsize=(20,8))
        # plt.plot(hourly_mean_df[hourly_mean_df["Column"] == col_list[i]][per],hourly_mean_df[hourly_mean_df["Column"] == col_list[i]]["Value"])
        # plt.title(col_list[i])
        # plt.grid()
        # plt.show()
    


# Prophet Model
generated_pdf.ln(20)
generated_pdf.set_font("Arial",'BI',16)
generated_pdf.set_fill_color(169,169,169)
generated_pdf.cell(0,10,txt="Forecasted Data ",ln=1,align="C",border=True,fill=True)
generated_pdf.ln(10)
generated_pdf.set_font("Arial",'B',12)

for col in most_corr_columns[:n]:
    print("Analysis for column: ",col)
    prophet_df, prophet_forecast, prophet_forecast_predictions = prophet_model(daily_grouped_df,col)

    #Create jump forecasts
    forecast_string_10_jump = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=10,type="Jump")
    forecast_string_25_jump = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=25,type="Jump")
    forecast_string_50_jump = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=50,type="Jump")
    forecast_string_75_jump = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=75,type="Jump")
    forecast_string_100_jump = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=100,type="Jump")
    if (forecast_string_10_jump != None):
        generated_pdf.cell(0,10,txt=forecast_string_10_jump,align="L",ln=1)
    if (forecast_string_25_jump != None):
        generated_pdf.cell(0,10,txt=forecast_string_25_jump,align="L",ln=1)
    if (forecast_string_50_jump != None):
        generated_pdf.cell(0,10,txt=forecast_string_50_jump,align="L",ln=1)
    if (forecast_string_75_jump != None):
        generated_pdf.cell(0,10,txt=forecast_string_75_jump,align="L",ln=1)
    if (forecast_string_100_jump != None):
        generated_pdf.cell(0,10,txt=forecast_string_100_jump,align="L",ln=1)

    #Create drop forecasts
    forecast_string_10_drop = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=10,type="Drop")
    forecast_string_25_drop = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=25,type="Drop")
    forecast_string_50_drop = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=50,type="Drop")
    forecast_string_75_drop = forecasted_strings(prophet_df, prophet_forecast_predictions,col,percent=75,type="Drop")
    if (forecast_string_10_drop != None):
        generated_pdf.cell(0,10,txt=forecast_string_10_drop,align="L",ln=1)
    if (forecast_string_25_drop != None):
        generated_pdf.cell(0,10,txt=forecast_string_25_drop,align="L",ln=1)
    if (forecast_string_50_drop != None):
        generated_pdf.cell(0,10,txt=forecast_string_50_drop,align="L",ln=1)
    if (forecast_string_75_drop != None):
        generated_pdf.cell(0,10,txt=forecast_string_75_drop,align="L",ln=1)


#---------------------------------------------------Supervised Data-----------------------------------------------------#

#Plot values of categorical column
if ("target" in df.columns or "output" in df.columns):
    df.rename(columns={df.columns[-1] : "target"},inplace=true)  
    output_count = df.iloc[:,-1].value_counts()[:5]
    print("The top most frequent output type is:\n",output_count)
    # plt.plot(output_count.index,output_count)
    # plt.grid()
    # plt.show()

    n=5
    imp_columns = column_feature_importance_target(df,num_columns,"Temperature")
    important_columns = imp_columns["feature"][:n].values
    print("The important columns are: ",important_columns)


#Output generated PDF
generated_pdf.output("data_report.pdf",'F')


#-------------------------------------------------------------------Plots------------------------------------------------#
plots_pdf = FPDF()
plots_pdf.add_page()
plots_pdf.set_font("Arial",'BI',16)
plots_pdf.cell(0,10,txt="Plots",ln=1,align="C",border=True)
# plots_pdf.set_auto_page_break(0)

for col in most_corr_columns:
    plt.figure(figsize=(12,4),facecolor='black')
    plt.plot(df["date"],df[col])
    plt.plot(df["date"],df[col].rolling(round(0.08*df.shape[0])).mean(),label=str(round(0.08*df.shape[0]))+"MA")
    plt.plot(df["date"],df[col].rolling(round(0.25*df.shape[0])).mean(),label=str(round(0.25*df.shape[0]))+"MA")
    plt.title(col)
    plt.legend()
    plt.grid()
    plt.savefig("plots/general"+col,bbox_inches='tight')
    plt.show()

#Moving Averages
plt.figure(figsize=(12,4),facecolor='black')
for col in most_corr_columns:  
    plt.plot(df["date"],np.log(df[col].rolling(round(0.1*df.shape[0])).mean()),label=col)

plt.title("Moving Averages")
plt.legend()
plt.grid()
plt.savefig("plots/ma",bbox_inches='tight')
plt.show()

corr_df = df.corr(method="pearson")
plt.figure(figsize=(((8/5)*df.shape[1]),((6/5)*df.shape[1])),facecolor='black')
sns.heatmap(corr_df,annot=True,lw=2)
plt.title("Correlation Plot")
corr_plot = plt.savefig("plots/corr_plot",bbox_inches = 'tight')
plt.show()


img_list = [x for x in os.listdir("plots")]
# img_list = sorted(img_list,key = os.path.getmtime)

for img in img_list[::-1]:
    if (img.endswith('.png')):
        image = "plots\\"+img
        plots_pdf.image(image,w=190,h=100)
        plots_pdf.ln(8)

plots_pdf.output("plots/plots.pdf")

end = time.time()
print("Total Execution Time: ",((end-start)/60)," minutes")