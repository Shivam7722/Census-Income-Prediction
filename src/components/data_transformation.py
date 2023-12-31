import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be on-hot encoding and which should be scaled
            categorical_cols = ['occupation','workclass','marital-status','relationship','race','native-country', 'gender']
            numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss','hours-per-week']
            
            # Defiing custome ranking for each categorical column
            occupation_categories  = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty',
                   'Other-service', 'Sales', 'Craft-repair', 'Transport-moving',
                   'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv',
                   'Armed-Forces', 'Priv-house-serv']

            workclass_categories = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov',
                  'Self-emp-inc', 'Without-pay', 'Never-worked']

            marital_status_categories = ['Never-married', 'Married-civ-spouse', 'Divorced',
                       'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']

            relationship_categories = ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']

            race_categories = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']   

            country_categories = ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'South',
                     'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
                     'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
                     'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic',
                     'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia',
                     'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                     'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
                     'Holand-Netherlands']
            
            gender_categories = ['Male', 'Female'] 
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ])

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories = [occupation_categories, workclass_categories, marital_status_categories, 
                                                         relationship_categories, race_categories, country_categories, gender_categories])),
                ('scaler', StandardScaler())
                ])
            

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols),
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            train_df.replace('?',np.nan ,inplace=True)
            test_df.replace('?',np.nan ,inplace=True)

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'income'
            drop_columns = [target_column_name,'education']
            
            logging.info(f'Train Dataframe Head : \n{train_df}')


            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasforming using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)