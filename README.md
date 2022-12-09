# Data-Analysis-Tool
Data Analysis tool developed in Python to analyze and forecast multivariate time series input data. This tool performs time series analysis and forecasts data values in the near future.

<hr>

> <b> NOTE: Download the required packages and dependencies [here](/requirements.txt). <br> 

## Overview 

This generalized data analysis tool is designed to work on unseen input data. However, there are some rules that the data must abide by before being fed into the script. 
<br>

## Data Requirements
The dataset used to develop and test the script can be found [here](/weather_data_denver.csv). This data has <b>145k</b> rows and <b>11</b> features. <br>
Here is a snapshot of the same data to understand what the input data should look like.
![image](https://user-images.githubusercontent.com/72503778/205184189-a46eca1e-3584-4027-b91c-63d7e24fee24.png)

<h3>Following are brief rules that tell what the data should look like: </h3>

1) The first column has to be DateTime. 
2) If date and time are together in one column, the format should be `"dd-mm-YY hh:mm:ss"`
4) The time should be in 24-hour format
5) The data should be time series.
6) The dependent variable (if any) should be the last column of the dataset and should have the name `"output"` or `"target"`.
7) The data should be continuous (should not miss any dates).
8) Ideally, the data should not contain any missing values.

## Loading data
<h3>There are two ways to load data into the model: </h3>

1) <b> Directly call CSV file using `pandas`: </b>
- Add the intended CSV file to the project directory
- On line `438` of the code, change the string variable `data_name` and set it to the name of the CSV file intended
- Run the code and look for `"Data Successfully Added!"` message in the terminal.
> In this case, all rows of the dataset will be called. To control the number of rows fetched, use the second method to load data.

2) <b> Fetch data from SQL Database: </b>
- Add the CSV file to the SQL server database. You can find the credentials for the server [here](/credentials.txt).
- On lines `434` through `436` of the code, change string variables according to the schema and table name in the database.
- Run the code and look for `"Data Successfully Added!"` message in the terminal.
  
By the time data is loaded and you see the `"Data Successfully Loaded!"` message, the program has already started working on the dataset.

## Key Points
 - This program extracts the important columns in the dataset based on their absolute correlation. By default, it will extract top 5 columns and work on them.
 - To change this number, navigate to the line `527` of the code, and change the value of integer variable `n` as required.
 - However, incrementing this number will increase the net execution time of the program.

<br>
  
This program thoroughly analyzes the numeric features of the time series data and generates a report in PDF format. Namely two PDFs, `plots.pdf` and `data_report.pdf`.
> The plots and combined `plots.pdf` will be saved in the plots subdirectory.