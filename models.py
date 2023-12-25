from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV,KFold
from sklearn.metrics import accuracy_score

import joblib
import os

class Models():
    
    def __init__(self,):
        self.models = {
            'rf' : RandomForestClassifier,
            'tree' : DecisionTreeClassifier,
            'Logis' : LogisticRegression,
            #TODO: FIX THIS ERROR!! 
            """
            ERROR : 
                return hasattr(X, "flags") and X.flags.c_contiguous
                                   ^^^^^^^^^^^^^^^^^^^^
                AttributeError: 'Flags' object has no attribute 'c_contiguous'
            """
            # 'knn' : KNeighborsClassifier,
            'svm' : SVC
        }
        self.params = {
            'rf' : {
                'max_depth' : [3,4,5],
                'n_estimators' : [200,300],
                'max_features' : [3,4],
                'ccp_alpha' : [0.05]
            },
            'tree' : {
                'max_depth' : [3,4,5],
                'max_features' : [3,5,6],
                'random_state' : [42],
            },
            'Logis' : {
                'solver' : ['lbfgs'],
                'penalty' : ['l1','l2'],
                'c' : [0.01,0.05]
            },
            'knn' : {
                'n_neighbors' : [1,2,3,4,5],
                'metric' : ['euclidean','manhattan'],
            },
            'svm' : {
                'C' : [0.01,0.05]
            }
            
        }
        
    def best_model(self,training_data):
        
        best_model = None 
        best_score = 0
        best_name_model = ""
        
        X_train,X_test,y_train,y_test = training_data
        
        results = []
        
        for n,m in self.models.items():
            print(f"training model ... {n}")
            
            # if n == 'knn':
            #     model = m(n_neighbors=2).fit(X_train,y_train)
            # else:
            model = m().fit(X_train,y_train)
                
            y_predict = model.predict(X_test)
            acc = accuracy_score(y_test,y_predict)
            
            results.append({n : acc})
            
            if acc > best_score:
                best_score = acc
                best_model = model
                best_name_model = n
                 
        print(results)
        return best_model,best_name_model
        
        
    def random_training(self,best_model,name_model,training_data):
        
        print(f"randomized search to model {name_model}")
        
        X_train,X_test,y_train,y_test = training_data
        
        kfold = KFold(n_splits=4,random_state=42,shuffle=True)
        
        random = RandomizedSearchCV(best_model,
                                  self.params[name_model],
                                  cv=kfold,
                                  scoring='accuracy'
                                  ).fit(X_train,y_train)
        
        y_predict = random.predict(X_test)
        print(f"accuracy is : {accuracy_score(y_test,y_predict)}")
        
        ##save model
        if not os.path.exists("models_folder"):
            os.mkdir("models_folder")
            
        joblib.dump(random.best_estimator_,"./models_folder/model.pkl")
        