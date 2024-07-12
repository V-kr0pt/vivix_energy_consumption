from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class Preprocess:
    def __init__(self, data, numerical_features, categorical_features, boolean_features, target):
        self.data = data
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.boolean_features = boolean_features
        self.target = target

        self.features = self.numerical_features + self.categorical_features + self.boolean_features
    
    def create_lag_columns(self, lag_columns, lag_values, data=None):
        if data is None:
            data = self.data
        # Create lag columns
        for column, value in zip(lag_columns, lag_values):
            lag_column_name = column+f'_lag{value}'
            data[lag_column_name] = data[column].shift(value)

            if column in self.numerical_features or column == self.target:
                self.numerical_features.append(lag_column_name)
            elif column in self.categorical_features:
                self.categorical_features.append(lag_column_name)
            elif column in self.boolean_features:
                self.boolean_features.append(lag_column_name)
        
        # update features
        self.features = self.numerical_features + self.categorical_features + self.boolean_features 

        return data 
    
    def create_preprocessor(self, imputer_stategy=None, scale_std=False, scale_minmax=False):

        # Transformer to numerical columns
        if imputer_stategy is None:
            step = []
        else:
            step = [('imputer', SimpleImputer(strategy='mean'))] # If there are missing values, fill with the mean (Maybe change because of the lag columns)

        # Add scaler to the pipeline
        if scale_std and scale_minmax:
            raise ValueError('Only one scaler can be selected')
        elif scale_std:
            step.append(('scaler', StandardScaler()))
        elif scale_minmax:
            step.append(('scaler', MinMaxScaler()))

        if step != []:
            numeric_transformer = Pipeline(steps=step) # Pipeline to numerical columns
        else:
            numeric_transformer = 'passthrough'

        # Transformer to categorical columns
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Transformers pipeline
        transformers = [
            ('num', numeric_transformer, self.numerical_features),
            ('cat', categorical_transformer, self.categorical_features),
            ('bool', 'passthrough', self.boolean_features)
        ]

        # Create the preprocessor
        preprocessor = ColumnTransformer(transformers=transformers)

        return preprocessor