import pandas as pd
from sklearn.model_selection import train_test_split

from dsna.automl.utils.operate import Transform
from dsna.automl.preservice.VIP.train import CreateModel
from dsna.automl.preservice.VIP.test import Test


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
    
class Apply(CreateModel, Test):

    def apply_sdv_model(self, models_list, real_data, sample_size):
        
        sample_dict = {}

        for model in models_list:

            model_name, parameters = self.process_models_list(model[0])
            algorithm = self.format_string_with_num(model_name)

            sdv_model = self.create_sdv_model(algorithm , parameters)

            sdv_model.fit(real_data)

            sample_data = sdv_model.sample(sample_size)

            evaluation_result = self.test_sdv_values(sample_data, real_data)

            sample_dict[model_name] = {'Data':sample_data,
                                       'Evaluation':evaluation_result}

        return sample_dict