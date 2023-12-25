import pandas as pd 
import numpy as np

from utils import Utils

import os

if __name__ == "__main__":
    
    utils = Utils()
    
    df = utils.pipeline_process_data()
    smote = False
    
    if not os.path.exists("models_folder/model.pkl"):
        model = utils.pipeline_training(df,smote)
    
    
        
    
    

