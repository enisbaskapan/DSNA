import pandas as pd
from sklearn.model_selection import train_test_split

from automl.preservice.VIP.train import CreateModel
from automl.preservice.VIP.test import Test


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