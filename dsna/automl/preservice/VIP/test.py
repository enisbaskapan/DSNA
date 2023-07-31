import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score, classification_report

from dsna.automl.utils.operate import Process

class Test:
    ''' Calculates different metrics (percentage error, mean squared error, R2 ...) '''
    
    def test_percentage_error(self, error_df):
        ''' Returns a list of percentage error values '''
        
        # Create error list
        percentage_error_list = []
        
        # Iterate over rows to calculate PE
        for index, row in error_df.iterrows():
            percentage_error = (abs(row[error_df.columns[0]] - row[error_df.columns[1]]) / row[error_df.columns[0]]) * 100
            percentage_error_list.append(percentage_error)
            
        return percentage_error_list
    
    def test_error_values(self, error_df):
        ''' Return a list of error metrics '''
        
        mepe = round(error_df['error(%)'].median(), 3)
        mpe = round(error_df['error(%)'].mean(), 3)

        meae = round(median_absolute_error(error_df['true_values'], error_df['predictions']), 3)
        mae = round(mean_absolute_error(error_df['true_values'], error_df['predictions']), 3)
        mse = round(mean_squared_error(error_df['true_values'], error_df['predictions']), 3)
        rmse = round(np.sqrt(mse), 3)
        normalized_rmse = rmse/(error_df['true_values'].max()-error_df['true_values'].min())

        std = round(error_df['true_values'].std(), 3)

        error_list = [mepe, mpe, meae, mae, mse, rmse, normalized_rmse, std]

        return error_list
    
    def test_variability_values(self, error_df, X_test):
        ''' Returns a list of variability values '''
    
        # R2 Score
        r2 = r2_score(error_df['true_values'], error_df['predictions'])
        
        # Adj R2 ---> 1-(1-R2)*(n-1)/(n-p)
        adj_r2 = 1-((1-r2) * ((X_test.shape[0]-1) / (X_test.shape[0] - X_test.shape[1]-1)))
        
        # Explained Varience Score
        evs = explained_variance_score(error_df['true_values'], error_df['predictions'])

        variability_values_list = [r2, adj_r2, evs]

        return variability_values_list
    
    def test_classification_report(self, y_true, y_pred):
        ''' Returns a dictionary of classification report results'''
        classification_report_dictionary = classification_report(y_true, y_pred, output_dict=True)
        report_list = list(classification_report_dictionary.keys())
        report_classes_list = report_list[:-3]
        accuracy = classification_report_dictionary['accuracy']

        for class_ in report_classes_list: classification_report_dictionary[class_]['accuracy'] = accuracy

        return classification_report_dictionary
    
class AssembleTest(Test, Process):
    ''' Create dataframse with test values '''
    
    def assemble_test_tables(self, regression_test_dict):
        ''' Create a dataframe with y, prediction and percentage error values and add it to dictionary'''
        
        test_tables_dict = {}
        for key, value in regression_test_dict['predictions'].items():

            error_df = pd.DataFrame()
            data_name = self.process_data_string(key)
            y_true = regression_test_dict['y_test'][data_name]
            error_df['true_values'] = y_true
            error_df['predictions'] = value

            # Retrieve percentage error values for each model
            error_df['error(%)'] = self.test_percentage_error(error_df)

            test_tables_dict[key] = error_df
        return test_tables_dict
        
    def assemble_error_values(self, regression_test_dict):
        ''' Returns a dataframe with error metrics for each model '''
        
        error_lists = []
        model_name_list = []
        for key, value in regression_test_dict['test_tables'].items():

            error_list = self.test_error_values(value)
            error_lists.append(error_list)
            model_name_list.append(key)

        error_values_df = pd.DataFrame(error_lists, index=model_name_list, columns = ['MEPE','MPE','MEAE','MAE','MSE','RMSE', 'NRMSE', 'STD'])
        error_values_df.index.names = ['ALGORITHM']
 
        return error_values_df
    
    def assemble_variability_values(self, regression_test_dict):
        ''' Returns a dataframe with variability values for each model'''
        
        variability_values_lists = []   
        model_name_list = []
        for key, value in regression_test_dict['test_tables'].items():
            
            data_name = self.process_data_string(key)
            variability_values_list = self.test_variability_values(value, regression_test_dict['X_test'][data_name])  
            variability_values_lists.append(variability_values_list)
            model_name_list.append(key)

        variability_values_df = pd.DataFrame(variability_values_lists, index=model_name_list, columns = ['R2', 'R2^', 'EVS'])
        variability_values_df.index.names = ['ALGORITHM']
        return variability_values_df
    
    def assemble_classification_report(self, classification_test_dict):
        ''' Returns a dataframe with classification report values for each model '''
        
        classification_df= pd.DataFrame()

        for key, value in classification_test_dict['predictions'].items():
            data_name = self.process_data_string(key)
            y_true = classification_test_dict['y_test'][data_name]
            y_pred = value
            
            classification_report_dictionary = self.test_classification_report(y_true, y_pred)
            classification_report_df = self.process_classification_report(classification_report_dictionary)
            classification_report_df['ALGORITHM'] = key
            classification_report_df.set_index('ALGORITHM', inplace=True)
            classification_df = pd.concat([classification_df, classification_report_df])
        classification_df.columns = classification_df.columns.str.upper()
        return classification_df