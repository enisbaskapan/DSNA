from math import ceil
import string

class CreateRange:

    def create_numerical_ranges(self, numeric_variable, thresholds_list):
        ''' Create a list of ranges for numeric variable you want to categorize

            Parameters
            ----------
            numeric_variable: Series
                        Column of a DataFrame that contains numerical values that you want to categorize
            thresholds_list: List
                        List of the thresholds 
        '''
        numerical_ranges_list = []
        start = int(numeric_variable.min()-1)
        final = int(numeric_variable.max()*1.5)
        for threshold in thresholds_list:
            numerical_ranges_list.append(range(start, threshold))
            start = threshold

        numerical_ranges_list.append(range(threshold, final))

        return numerical_ranges_list

    def create_categorical_ranges(self, interval,start,finish):
        ''' Create a list of ranges for categorcial variable you want to categorzie

            Parameters
            ----------
            interval: int
                Count of intervals to create
            start: int -- float
                minnimum value
            finish: int -- float
                maximum value
        '''
        categorical_ranges_list = []
        start = int(start)
        finish = int(finish)
        diff = int((finish - start) / interval)

        for i in range(interval):
            if interval-i != 1:
                categorical_ranges_list.append(range(start, start+ diff))
            else:
                categorical_ranges_list.append(range(start, ceil(1.5 * (start+diff))))

            start += diff

        return categorical_ranges_list
    
class Categorize:
    
    def __init__(self):
        self.letters = string.ascii_letters
    
    def categorize_numerical_variable(self, df_series, numerical_ranges_list):
        ''' Returns a dictionary of categories with corresponding ranges 
        
            Parameters
            ----------
            df_series: Series
                    Numerical variable series
            numerical_ranges_list: List
                    List of ranges for numerical values
        '''
        
        letters = self.letters[:len(numerical_ranges_list)]
        range_dict = dict(zip(letters, numerical_ranges_list))
        category_dict = {}

        for key, value in range_dict.items():
            for data in  df_series.values:
                if int(data) in value:
                    category_dict[data] = key

        return category_dict
    
    def categorize_categorical_variable(self, grouped_data, categorical_ranges_list):
        ''' Returns a dictionary of categories with corresponding ranges 

            Parameters
            ----------
            grouped_data: Series
                    Series acquired after using groupby method
            categorical_ranges_list: List
                    List of ranges for categorical values
        '''

        letters = string.ascii_letters[:len(categorical_ranges_list)]
        range_dict = dict(zip(letters, categorical_ranges_list))
        category_dict = {}
        for name, data in enumerate(grouped_data):
            for key, value in range_dict.items():
                if int(data) in value:
                    category_dict[grouped_data.index[name]] = key

        return category_dict
    
class DetectOutlier:
    pass

class AnalyzeError:
    pass