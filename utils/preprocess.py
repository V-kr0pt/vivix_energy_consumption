from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

class Preprocess:
    def __init__(self, data, numerical_features, categorical_features, boolean_features, target):
        self.data = data
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.boolean_features = boolean_features
        self.target = target

        self.features = self.numerical_features + self.categorical_features + self.boolean_features
    
    def create_lag_columns(self, lag_columns, lag_values, data=None, update_features=True):
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
        
        # update features if data is the train data
        if update_features:
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
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

        # Transformers pipeline
        transformers = [
            ('num', numeric_transformer, self.numerical_features),
            ('cat', categorical_transformer, self.categorical_features),
            ('bool', 'passthrough', self.boolean_features)
        ]

        # Create the preprocessor
        self.preprocessor = ColumnTransformer(transformers=transformers)

        #return preprocessor

    def fit(self,data):
        # Fit the preprocessor
        self.preprocessor = self.preprocessor.fit(data)

        # Update self.categorical_features with the transformed features
        self.categorical_features = list(self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())
        self.features = self.preprocessor.get_feature_names_out()
        self.features = self.features.tolist()
    
    def transform(self, data):
        assert self.preprocessor is not None, 'You need to fit the preprocessor before transforming the data'
        return self.preprocessor.transform(data) 
        #assert self.features == transformed_data.columns.tolist(), 'The columns in the data are different from the features used to fit the preprocessor'
               

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def save_preprocessor(self, path):
        joblib.dump(self.preprocessor, path)
    
    # This function returns the real boost power based on the glass color 
    def boost_power(self, color, boost_extraction):
        if color == 'incolor' or color == 0:
            return (boost_extraction*0.03)
        elif color == 'verde' or color == 1:
            return (boost_extraction*0.0302) + 1.3852
        elif color == 'cinza' or color == 2:
            return (boost_extraction*0.0296) + 1.0935

    