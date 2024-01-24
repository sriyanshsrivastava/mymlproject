import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation on the data
        '''
        try:
            numerical_column = ['writing_score','reading_score']
            categorical_column = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                "lunch",
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]
            )

            cat_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns;{categorical_column}')
            logging.info(f'Numerical columns;{numerical_column}')


            preprocessor = ColumnTransformer(
                [
                    ('num_pipline', num_pipeline, numerical_column),
                    ('cat_pipline', cat_pipline, categorical_column)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('data transformation initiated')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('read train and test data completed')
            logging.info('Obtaining preprocessor object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'

            numerical_column = ['writing_score','reading_score']

            input_feature_train_dataset = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_dataset = train_df[target_column_name]

            input_feature_test_dataset = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_dataset = test_df[target_column_name]

            logging.info('applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_dataset)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_dataset)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_dataset)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_dataset)]

            logging.info('saved Preprocessing object')
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )

        except Exception as e:
            raise CustomException(e,sys)





