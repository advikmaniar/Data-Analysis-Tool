from io import StringIO
import mysql.connector
from sqlalchemy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy.sql.base import InPlaceGenerative
from sqlalchemy.sql.schema import RETAIN_SCHEMA
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import time


# Write entire code inside try-except block to prevent errors from appearing on the streamlit dashboard.
try:

    start_time = time.time()
    df = pd.DataFrame()

    st.markdown("<h1 style='text-align: center; color: white;'>NASA Data Analysis Tool</h1>",
                unsafe_allow_html=True)
    st.sidebar.title("Sidebar")

    #---------------------------------USING DIRECT CSV FILES----------------------------------#
    def useDroppedFile(DROPPED_FILE):
        if DROPPED_FILE is not None:
            dataframe = pd.read_csv(DROPPED_FILE)
        return dataframe
    #-----------------------------------------------------------------------------------------#

    #--------------------------------EXTRACTING FILE FROM SQL DB------------------------------#
    # Establish Connection with SQL Database
    def connectDB(host, username, password):

        mydb = mysql.connector.connect(
            host=host,
            username=username,
            password=password
        )
        return mydb
    #-----------------------------------------------------------------------------------------#

    def dataPull(data_pull_method):

        global df
        if (data_pull_method == "Drop CSV File"):
            try:
                DROPPED_FILE = st.file_uploader(label="Drop CSV file here")
                df = useDroppedFile(DROPPED_FILE)
            except:
                st.info("Please load a file to continue...")

            return df

        elif (data_pull_method == "Pull data from SQL Database"):
            try:
                st.header("Connect your SQL Database here")
                host_input = st.text_input(
                    "Please enter the host name:", placeholder="Host")
                username_input = st.text_input(
                    "Please enter your username:", placeholder="Username")
                password_input = st.text_input(
                    "Please enter your password:", placeholder="Password", type="password")

                mydb = connectDB(host_input, username_input, password_input)
                st.write("Successfully connected to the server!")

                st.info(
                    "Enter the schema and table name of the data you want to pull")
                SCHEMA_USED = st.text_input(
                    "Schema: ", placeholder="Schema") + "."
                DATA_USED = st.text_input("Table Name: ", placeholder="Table")
                # Write a SELECT query to fetch data from the DB
                FETCH_QUERY = "SELECT * FROM "+SCHEMA_USED+DATA_USED

                # Pull data from DB using pandas
                df = pd.read_sql(FETCH_QUERY, mydb)

            except:
                st.info("Please complete the above fields to continue...")

            return df

        else:
            st.info("Please select an option from the sidebar.")
            return None

    # Display dataset on dashboard
    data_pull_method = st.sidebar.selectbox(
        "How do you want to pull your data? ", ("Drop CSV File", "Pull data from SQL Database"), index=0)
    df = dataPull(data_pull_method)

    def preprocessData(df):

        # Drop null entries
        df.dropna(how="all")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    preprocessData(df)

    def describeData(df):
        st.write("Data Used: ", df.head())
        st.write("Number of entries: ", df.shape[0])
        st.write("Number of columns: ", df.shape[1])
        if "date" in df.columns:
            first_date = df["date"].min()
            last_date = df["date"].max()
            st.write("This data ranges from ", first_date, " to ", last_date)

    describeData(df)

    st.markdown("""---""")

    def column_specific_data(df,column_name):

        description = df.describe()
        st.subheader(column_name)
        st.write("Mean Value: ", round(description[column_name][1], 4), ", Minimum Value: ", round(
            description[column_name][3], 4), ", Maximum Value: ", round(description[column_name][7], 4))
        column_plot = plt.bar(df.date,df[column_name])
        plt.grid()
        plt.show()
        plt.xticks(rotation=45)
        st.write(column_plot)
        st.pyplot()


    selected_column = st.sidebar.selectbox(
        "Choose a column to view data. ", df.columns, index=1)
    column_specific_data(df,selected_column)

    end_time = time.time()
    st.info(
        f"Total execution time: {round((end_time - start_time),4)} seconds")

except:
    st.info("...................................................................................................................." +
            "....................................................")
    # st.error("There was some issue with the data....")
