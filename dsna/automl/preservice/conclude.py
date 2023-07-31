import pandas as pd
from sklearn.model_selection import train_test_split

from dsna.automl.utils.operate import Transform

class Preprocess(Transform):
    
    def preprocess_test_data(self, data, dependent_variable):
        

        X = data.drop(dependent_variable, axis=1)
        y = data[dependent_variable]
        
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_test = self.transform_test_data(X_train, X_test)


        return  X_train, X_test, y_train, y_test
        
    def preprocess_data(self, data, dependent_variable):
        

        X = data.drop(dependent_variable, axis=1)
        y = data[dependent_variable]
        
        X = pd.get_dummies(X, drop_first=True)
        
        X, scaler = self.transform_data(X)

        return X, y, scaler
    
    def preprocess_time_series_data(self, data, test_size):
        
        train, test = train_test_split(data, test_size=test_size)
        
        return train, test