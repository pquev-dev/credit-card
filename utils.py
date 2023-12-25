import yaml 
import pandas as pd
import numpy as np
import joblib
from models import Models

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomunderSampler

import os

class Utils():
    
    def __init__(self,):
        self.config = "./main.yaml"
        with open(self.config,'r') as cfg:
            self.config = yaml.safe_load(cfg)
        self.csv_path = self.config['data']['raw']
        self.total_amplify = self.config['data']['amplify']
        self.model_class = Models()
    
    #load csv 
    def load_csv(self,path):
        try:
            return pd.read_csv(path)
        except Exception as err:
            print(err)
            
    # amplify all dataset
    def amplify(self,df,total):
        return pd.DataFrame(np.repeat(df.values,total,axis=0),columns=df.columns)
    
    #crear nulls values
    def clear_nulls(self,df):
        if len(df) !=  0 :
            return df.dropna()
        
    def pipeline_process_data(self,):
        df = self.load_csv(self.csv_path)        
        df = self.amplify(df,self.total_amplify)
        df = self.clear_nulls(df)
        return df
    
    def pipeline_training(self,df,smote=False):
        
        print(f"total class : {df['Class'].value_counts()}")
        
        X = df.drop([self.config['target']],axis=1)
        y = df[self.config['target']]
        
        #50% 1 50% 0
        if smote:
            rus = RandomunderSampler(sampling_strategy=1)
            X,y = rus.fit_resample(X,y)
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=self.config['test_size'],random_state=42)
        training_data = (X_train,X_test,y_train,y_test)
        
        best_model,name = self.model_class.best_model(training_data)
        self.model_class.random_training(best_model,name,training_data)
        

    def save_model(self,model,path):
        try:
            if not os.path.exists(path):
                os.mkdir(path)
                joblib.dump(model,path)
        except Exception as err:
            print(err)
        finally:
            print(f"saved model :))))")
        
        
        
        
        

        
    
        