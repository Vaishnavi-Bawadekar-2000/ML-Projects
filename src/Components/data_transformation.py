
import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.append(PACKAGE_ROOT)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationCnfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationCnfig()

    def get_data_transformation_object(self):

        try:
             numerical_col = ["reading_score","writing_score"]
             categorical_col=["gender","race_ethnicity","parental_level_of_education","lunch", "test_preparation_course" 
                              ]
             
             num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                 ]
              )
             
             cat_pipeline = Pipeline(
                  steps=[
                      ("imputer" , SimpleImputer(strategy="most_frequent")),
                     ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse=False)),
                  ]
                )  

             logging.info(f"Numerical columns:{numerical_col} ")
             logging.info(f"categorical columns:{categorical_col} ")

             preprocessor = ColumnTransformer(
                 [
                 ("num_pipeline", num_pipeline, numerical_col),
                 ("cat_pipeline", cat_pipeline, categorical_col)
                 ]
             )
             return preprocessor
        except Exception as e:
            raise CustomException(e,sys)    
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
             train_df = pd.read_csv(train_path)
             test_df = pd.read_csv(test_path)

             logging.info("Read train and test data completed")

             logging.info("Obtaining preprocessing object")

             preprocessing_obj= self.get_data_transformation_object()

             target_col_name="math_score"
             numerical_col = ["reading_score","writing_score"]

             input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
             traget_feature_train_df = train_df[target_col_name]

             input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
             traget_feature_test_df = test_df[target_col_name]

             logging.info(f"applying preprocessing train and test dataframe")

             input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
             input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
             

             train_arr= np.c_[input_feature_train_arr, np.array(traget_feature_train_df)]
             test_arr= np.c_[input_feature_test_arr, np.array(traget_feature_test_df)]
             

             logging.info(f"saved the preprocessing object")
              
             save_object(
                   self.data_transformation_config.preprocessor_obj_file_path,
                   obj= preprocessing_obj
              ) 
             

             return(
                 train_arr,
                 test_arr,
                 self.data_transformation_config.preprocessor_obj_file_path
             )
        except Exception as e:
            raise CustomException(e,sys)
