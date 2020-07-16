import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    df = pd.merge(df, ndc_df[['Non-proprietary Name', 'NDC_Code']],left_on='ndc_code', right_on='NDC_Code')
    #change the column name to generic_drug_name
    df.rename(columns={'Non-proprietary Name':'generic_drug_name'}, inplace=True)
    return df

#Question 4
def select_first_encounter(df, patient_id, encounter_id):
    temp = df.sort_values(encounter_id)
    temp.reset_index(drop=True,inplace=True)
    first_encounter_df = temp.drop_duplicates(subset=encounter_id, keep='first')
    first_encounter_df = first_encounter_df.drop_duplicates(subset=patient_id, keep='first')
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    test_percentage=0.2
    
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    sample_size = int(total_values - 2*round(total_values * test_percentage))
    test_size = int((total_values - sample_size)/2)
    train_values = unique_values[:sample_size]
    temp_values = unique_values[sample_size:]
    test_values = temp_values[:test_size]
    val_values = temp_values[test_size:]
    train = df[df[patient_key].isin(train_values)].reset_index(drop=True)
    validation = df[df[patient_key].isin(test_values)].reset_index(drop=True)
    test = df[df[patient_key].isin(val_values)].reset_index(drop=True)
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = '?'
    s = '?'
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    return student_binary_prediction
