import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Input, Dense

class Process:
    
    def process_data_string(self, data_string):
        """
        Reformat data_string to store data
        """

        # Retrieve indexes of digits inside the data string
        indexes = [index for index, letter in enumerate(data_string) if letter.isdigit()]

        for index, index_value in enumerate(indexes):

            # Assign the value on the previous index
            previous_index_value = indexes[index - 1]

            if index > 0:

                # Compare index values that are in order to find the cut between data and algorithm strings
                if previous_index_value + 1 != index_value:

                    # Find the cut then break! 
                    # Otherwise would be retrieving the entire string or get and error

                    end_index_of_indexes = indexes.index(index_value)- 1
                    end_index_of_data_string = indexes[end_index_of_indexes] + 1
                    break

        return data_string[:end_index_of_data_string]
    
    
    def process_classification_report(self, classification_report_dictionary):
        """
        Reformat classification report dictionary to store result in a dataframe
        """
        del classification_report_dictionary['accuracy']
        del classification_report_dictionary['macro avg']
        del classification_report_dictionary['weighted avg']

        classification_report_df = pd.DataFrame.from_dict(classification_report_dictionary).transpose()
        classification_report_df = classification_report_df.reset_index()
        classification_report_df = classification_report_df.rename(columns={'index': 'class'})

        return classification_report_df
    
    
    def process_input_layer(self, layers, X_train):

        n_features = X_train.shape[1]
        input_layer = Input(shape=(n_features,))

        if type(input_layer) == type(layers[0]):

            units = [lyr.input_shape[1] for lyr in layers[1:]]
            units.append(1)

            layers = [Dense(unit, activation ='relu') for unit in units]
            del layers[0]

            layers.insert(0, input_layer)

        else:
            layers.insert(0, input_layer)

        return layers

    
    def process_prophet_data(self, data):
        
        data = data.reset_index()
        data.columns = ['ds','y']
        
        return data
    
    def process_prophet_index(self, train, test):
        
        train = train.set_index('ds')
        test = test.set_index('ds')
        
        return train, test
    
    def process_prophet_predictions(self, predictions):
        
        predictions = predictions[['ds', 'yhat']]
        predictions = predictions.set_index('ds')
        
        return predictions
    
    def process_models_list(self, model):
        
        return list(model.values())
    
class Transform:
    
    def transform_data(self, data, feature_range = (0,1)):
        
        scaler = MinMaxScaler(feature_range = feature_range)
        data_scaled = scaler.fit_transform(data)
        return data_scaled, scaler
        
    def transform_test_data(self, X_train, X_test):    
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test