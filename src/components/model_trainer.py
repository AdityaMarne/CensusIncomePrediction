import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import io
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from src.exception import CustomException
from sklearn.metrics import ConfusionMatrixDisplay
from src.logger import logging

from src.utils import save_object
from dataclasses import dataclass


@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Independent and dependent variables form train and test array')
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = LogisticRegression()
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)
            test_score = accuracy_score(y_test,y_pred)
    

            #print(f'Model Name: Logistic Regression, Accuracy Score: {test_score}')
            logging.info(f'Model Name: Logistic Regression, Accuracy Score: {test_score}')

            cm = confusion_matrix(y_test,y_pred)
            logging.info(f'Model Name: Logistic Regression, Confusion Matrix: {cm}')


            # # Create the ConfusionMatrixDisplay plot
            # disp = ConfusionMatrixDisplay(cm)
            # fig, ax = plt.subplots(figsize=(10, 8))
            # disp.plot(ax=ax)

            # # Save the plot to a bytes buffer
            # buf = io.BytesIO()
            # plt.savefig(buf, format='png')
            # buf.seek(0)

            # # Log the plot as a file
            # logging.info("Confusion Matrix Plot:")
            # logging.info(f"[CONFUSION_MATRIX_PLOT]{buf.getvalue()}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = model
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)