class Check:
    
    def check_if_size_is_smaller_than_two(self, size):
    
        if size <= 2:
            size_is_smaller_than_two = True

        else:
            size_is_smaller_than_two = False

        return size_is_smaller_than_two

    def check_if_element_is_in_data(self, element, data):

        if element in data:
            element_is_in_data = True
            
        else:
            element_is_in_data = False
            
        return element_is_in_data
    
    def check_if_elements_are_equal(self, first_element, second_element):
        
        if first_element == second_element:
            elements_are_equal = True
            
        else:
            elements_are_equal = False
        
        return elements_are_equal
        
    def check_if_there_are_multiple_words(self, string):
        
        word_count = len(string.split(' '))
        
        if word_count > 1:
            there_are_multiple_words = True
        
        else:
            there_are_multiple_words = False
            
        return there_are_multiple_words
        
    def check_if_there_are_more_than_two_elements(self, data):
        
        element_count = len(data)
        
        if element_count > 2:
            
            there_are_more_than_two_elements = True
        
        else:
            
            there_are_more_than_two_elements = False
            
        return there_are_more_than_two_elements
    
    def check_if_there_are_two_elements(self, data):
        
        element_count = len(data)
        
        if element_count == 2:
            
            there_are_two_elements = True
        
        else:
            
            there_are_two_elements = False
            
        return there_are_two_elements
    
    def check_if_there_is_one_element(self, data):
        
        element_count = len(data)
        
        if element_count == 1:
            
            there_is_one_element = True
        
        else:
            
            there_is_one_element = False
            
        return there_is_one_element
    
    def check_if_indexes_are_consecutive(self, first_index, second_index):

        if first_index + 1 == second_index:

            indexes_are_consecutive = True

        else:

            indexes_are_consecutive = False

        return indexes_are_consecutive
    
    def check_if_order_is_correct(self, first_index, second_index):
    
        if first_index < second_index:
            order_is_correct = True

        else:
            order_is_correct = False
            
        return order_is_correct

class Format:
    
    def format_passenger_string(self, passenger_string):
        
        formatted_passenger_string = passenger_string.split(',')[0]
        
        return formatted_passenger_string
    
    def format_date_to_string(self, date_object):

        formatted_date_string = date_object.strftime("%d-%m-%Y")
        
        return formatted_date_string

    def format_string_with_num(self, string_with_num):
        
        """ 
        Reformat algorithm_string to get algorithm name
        
        """
        
        digit_num = len([letter for letter in string_with_num if letter.isdigit()])
        string_without_num = string_with_num[:-digit_num]
        
        return string_without_num