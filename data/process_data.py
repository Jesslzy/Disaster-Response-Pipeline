import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load two datasets and merge together.
    
    :param messages_filepath: csv file path
    :param ategories_filepath: csv file path
    
    """
    
    # Read in two dfs
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge two dfs
    df = messages.merge(categories, on = 'id')
    
    return df


def clean_data(df):
    """
    Clean dataframe, isolate and attach 36 target categories.
    
    """
    
    # Cleaning category names
    categories = df['categories'].str.split(';', expand = True)
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2])
    
    # Renaming columns of categories df
    categories.columns = category_colnames
    
    # Convey category values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Drop original category column and concatenate with newly created categories df
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df = df.replace(2, 0)
    
    # Dropping duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    
    """
    Save df as a SQL database for ML pipeline.
    
    :param database_filename: SQL database file path
    
    """
    
    # Saving df as SQL database
    table = 'disaster_response'
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(table, engine, if_exists='replace', index=False)
    


def main():
    
    """
    Call to run through this python file.
    
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()