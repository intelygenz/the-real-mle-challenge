from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from typing import Dict, List
import re

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from typing import List, Any, Dict


# Get number of bathrooms from `bathrooms_text`
def num_bathroom_from_text(text):
    try:
        if isinstance(text, str):
            bath_num = text.split(" ")[0]
            return float(bath_num)
        else:
            return np.nan
    except ValueError:
        return np.nan

def array_binding(array: List[int | float], bins: List[int | float], labels: List[Any]):
    """
    Function to replicate the behabiour of pandas.cut() function but with numpy

    Parameters
    ----------
    values : array-like of length n_samples
        The input data to be binned
    
    bins : array-like of length n_bins + 1
        The bin edges, representing the intervals for binning.
    
    labels : list or array-like, optional, default=None
        Labels corresponding to the bins.
    
    Returns
    -------
    An array of the same shape as `values`, where each element corresponds to 
    the label of the bin in which that element falls.

    """

    bin_indices = np.digitize(array, bins)

    bin_indices = np.clip(bin_indices - 1, 0, len(bins) - 1)


    return np.array(labels)[bin_indices]

def preprocess_amenities_column(df: DataFrame) -> DataFrame:
    
    df['TV'] = df['amenities'].str.contains('TV')
    df['TV'] = df['TV'].astype(int)
    df['Internet'] = df['amenities'].str.contains('Internet')
    df['Internet'] = df['Internet'].astype(int)
    df['Air_conditioning'] = df['amenities'].str.contains('Air conditioning')
    df['Air_conditioning'] = df['Air_conditioning'].astype(int)
    df['Kitchen'] = df['amenities'].str.contains('Kitchen')
    df['Kitchen'] = df['Kitchen'].astype(int)
    df['Heating'] = df['amenities'].str.contains('Heating')
    df['Heating'] = df['Heating'].astype(int)
    df['Wifi'] = df['amenities'].str.contains('Wifi')
    df['Wifi'] = df['Wifi'].astype(int)
    df['Elevator'] = df['amenities'].str.contains('Elevator')
    df['Elevator'] = df['Elevator'].astype(int)
    df['Breakfast'] = df['amenities'].str.contains('Breakfast')
    df['Breakfast'] = df['Breakfast'].astype(int)

    df.drop('amenities', axis=1, inplace=True)
    
    return df


def find_categories(array, categories)-> Dict[str, int]:
    string = str(array)
    return {category: int(category in string) for category in categories}

# Same regex function
def apply_regex(text, pattern):
    match = pattern.search(text)
    return match.group(0) if match else None

# Custom transformer split a string by spaces and cast to float the first element
class StringToFloatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Dict[str, str] | str):
        # Specify which columns to apply the transformation to and the new name (optional)
        self.columns = columns

    def fit(self, X, y=None):
        # No fitting needed
        return self

    def transform(self, X: DataFrame | np.ndarray):
        X_copy = X.copy()
        if self.columns:
            if isinstance(X, DataFrame):
                if isinstance(self.columns, dict):
                    for old_name, new_name in self.columns.items():
                        X_copy[new_name] =list(map(num_bathroom_from_text, X_copy[old_name])) 
                else:
                    for col in self.columns:
                        X_copy[col] =list(map(num_bathroom_from_text, X_copy[col])) 
                self.out_cols = list(X_copy.columns)
            if isinstance(X, np.ndarray):
                if isinstance(self.columns, dict):
                    self.out_cols = list(self.columns.values())
                    for i in range(len(self.columns)):
                        X_copy[i] =list(map(num_bathroom_from_text, X_copy[i])) 
                else:
                    self.out_cols = self.columns
                    for i in self.columns:
                        X_copy[i] =list(map(num_bathroom_from_text, X_copy[i])) 

        return X_copy
    
    def get_feature_names_out(self, columns):
        return self.out_cols
    
# Custom transformer to parse to numeric the number of bathrooms 
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str] | str):
        # Specify which columns to select
        self.columns = columns

    def fit(self, X, y=None):
        # No fitting needed
        return self

    def transform(self, X: DataFrame):
        X_copy = X.copy()
        if self.columns:
            if isinstance(X_copy, DataFrame):
                X_copy = X_copy[self.columns if isinstance(self.columns, list) else [self.columns]]
                self.out_cols = X_copy.columns
                return X_copy
            else:
                raise ValueError("The data provided must be a pandas.Dataframe") 
        self.out_cols = X_copy.columns
        return X_copy
    
    def get_feature_names_out(self, columns):
        return self.columns
    
# Custom transformer to rename columns 
class ColumnRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Dict[str, str]):
        if not isinstance(columns, dict):
            raise  ValueError("The columns must be passed as a dict with the format {'old_key':'new_key'}") 
        # Specify which columns to rename
        self.columns = columns

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X: DataFrame):
        X_copy = X.copy()
        if self.columns:
            if isinstance(X_copy, DataFrame):
                X_copy.rename(columns=self.columns, inplace=True)
                self.out_cols = X_copy.columns
                return X_copy
            else:
                raise ValueError("The data provided must be a pandas.Dataframe") 
        self.out_cols = X_copy.columns
        return X_copy
    def get_feature_names_out(self, columns):
        return self.out_cols
    
# Custom transformer to drop NAs in columns or rows  
class DropNan(BaseEstimator, TransformerMixin):
    def __init__(self, axis: int=0):
        # Specify which axis to evaluate 
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X: DataFrame | np.ndarray):
        
        if isinstance(X, DataFrame):
            X_copy = X.copy()
            X_copy =  X_copy.dropna(axis=self.axis)
            return X_copy
        elif isinstance(X, np.ndarray):
            X_copy = X.copy()
            X_copy =  X_copy[~np.isnan(X_copy).any(axis=self.axis)]
            return X_copy
        else:
            raise ValueError("The data provided must be a pandas.Dataframe or np.array")
    
    def get_feature_names_out(self, columns):
        return columns
        
# Custom transformer to drop NAs in columns or rows  
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: str | List[str]):
        # Specify which axis to evaluate 
        self.columns = columns

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X: DataFrame):
        
        if isinstance(X, DataFrame) and isinstance(self.columns, (str, list)):
            X_copy = X.copy()
            X_copy.drop(columns=self.columns, inplace=True)
        else:
            raise ValueError("The data provided must be a pandas.Dataframe and columns must be a string or list of strings")
        self.cols_out = X_copy.columns
        return X_copy
    
    def get_feature_names_out(self, columns):
        return self.cols_out
        
# Custom transformer to cast string to int applying regexpatter
class StringToInt(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str] | str, patterns: List[str] | str):

        if type(columns) != type(patterns):
            raise ValueError("The columns and patters must have the same data type")
        elif isinstance(columns, list): 
            if len(columns) != len(patterns):
                raise ValueError("columns and patterns list must have the same leght")
        elif not isinstance(columns, str):
            raise ValueError("columnas and patters must be or a single string or a list of strings")
        
        self.columns = columns
        self.patterns = patterns

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self
    
    # Function that applies a regex pattern to a string and returns the match
    def _apply_regex(self, text, pattern):
        match = pattern.search(text)
        return match.group(0) if match else None

    def transform(self, X: DataFrame | npt.ArrayLike):
        
        X_copy = X.copy()
        if isinstance(X_copy, DataFrame):
            if isinstance(self.columns, str):
                comp_patter = re.compile(self.patterns)
                X_copy[self.columns] = list(map(lambda x: int(self._apply_regex(x, comp_patter)), X_copy[self.columns]))
            else:
                for col, pattern in zip(self.columns, self.patterns):
                    comp_patter = re.compile(pattern)
                    X_copy[col] = list(map(lambda x: int(self._apply_regex(x, comp_patter)), X_copy[col]))
            self.out_cols = X_copy.columns
            return X_copy
        elif isinstance(X_copy, np.ndarray):
            self.out_cols = self.columns
            if isinstance(self.columns, str):
                comp_patter = re.compile(self.patterns)
                return X_copy.apply(lambda x: int(self._apply_regex(x, comp_patter)))
            else:
                for i, pattern in enumerate(self.patterns):
                    comp_patter = re.compile(pattern)
                    X_copy[i] = list(map(lambda x: int(self._apply_regex(x, comp_patter)), X_copy[i]))
            return X_copy
        elif isinstance(X_copy, list):
            self.out_cols = self.columns
            if isinstance(self.columns, str):
                comp_patter = re.compile(self.patterns)
                return list(map(lambda x: int(self._apply_regex(x, comp_patter)), X_copy))
            else:
                for i, pattern in enumerate(self.patterns):
                    comp_patter = re.compile(pattern)
                    X_copy[i] = list(map(lambda x: int(self._apply_regex(x, comp_patter)), X_copy[i]))
            return X_copy          
    def get_feature_names_out(self, columns):
        return self.out_cols
        
# Custom transformer to filter rows of a pandas.Dataframe
class QueryFilter(BaseEstimator, TransformerMixin):
    def __init__(self, query_string: str):
        # Specify which axis to evaluate 
        self.query_string = query_string

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X: DataFrame):

        if self.query_string:
            X_copy = X.copy()
            if isinstance(X_copy, DataFrame):
                try:
                    X_copy.query(self.query_string, inplace=True)
                    self.out_cols = X_copy.columns
                    return X_copy
                except Exception as e:
                    ValueError(f"Error applying the query string: {str(e)}")
            else:
                raise ValueError("The data provided must be a pandas.Dataframe")
        self.out_cols = X.columns
        return X
    
    def get_feature_names_out(self, columns):
        return self.out_cols 

# Custom transformer to aaply pandas cut
class DiscretizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            columns: str | List[str],
            new_colnames: str | List[str],
            bins: List[float|int] | List[List[float|int]], 
            labels: List[float|int] | List[List[float|int]]
            ):
        
        # Validate the input parameters
        if isinstance(columns, list):
            if not isinstance(bins, list) or not isinstance(labels, list):
                raise ValueError("If 'columns' is a list, 'bins' and 'labels' must also be lists.")
            
            for bin, label in zip(bins, labels):
                if len(bin) != (len(label) +1):
                    raise ValueError("'bins' must have the same length as 'labels' + 1.")
            
            if len(columns) != len(bins) or len(columns) != len(labels) or len(columns) != len(new_colnames):
                raise ValueError("'columns', 'bins', 'labels' and 'new_colnames' must have the same length when 'columns' is a list.")

        if isinstance(columns, str):
            if not isinstance(bins, list) or not isinstance(labels, list):
                raise ValueError("If 'columns' is a string, 'bins' and 'labels' must be lists.")

            if len(bins) != (len(labels)+1):
                raise ValueError("'bins' must have the same length as 'labels' + 1.")
        
        self.columns = columns
        self.bins = bins
        self.labels = labels
        self.new_colnames = new_colnames
        

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X: DataFrame):
        X_copy = X.copy()
        if isinstance(X_copy, DataFrame):
            if isinstance(self.columns, str):
                X_copy[self.new_colnames] = pd.cut(X_copy[self.columns], bins=self.bins, labels=self.labels)
                
            if isinstance(self.columns, list):
                for col, new_name, bin, label in zip(self.columns, self.new_colnames, self.bins, self.labels):
                    X_copy[new_name] = pd.cut(X_copy[col], bins=bin, labels=label)
            self.out_cols = X_copy.columns
            return X_copy
        else:
            self.out_cols = self.new_colnames
            if isinstance(self.columns, str):
                return pd.cut(X_copy, bins=self.bins, labels=self.labels).to_numpy()
                
            if isinstance(self.columns, list):
                for i, bin, label in enumerate(zip(self.bins, self.labels)):
                    X_copy[i] = pd.cut(X_copy[i], bins=bin, labels=label).to_numpy()
            return X_copy               
    def get_feature_names_out(self, columns):
        return self.out_cols

# Custom transformer to parse to numeric the number of bathrooms 
class ArrayOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, categories: List[str]):
        # Specify which columns to select
        self.column = column
        self.categories = categories

    def fit(self, X, y=None):
        # No fitting needed
        return self

    def transform(self, X: DataFrame | np.ndarray):
        X_copy = X.copy()
        if isinstance(X_copy, DataFrame):
            if isinstance(self.column, str):

                cat_df = pd.DataFrame(X_copy[self.column].apply(lambda x: find_categories(x, self.categories)).to_list(), index=X_copy.index)
                X_copy = pd.concat([X_copy, cat_df], axis=1, ignore_index=False)
            self.out_cols = X_copy.columns
            return X_copy
        if isinstance(X_copy, np.ndarray | pd.Series):
            self.out_cols = self.categories
            return pd.DataFrame(list(map(lambda x: find_categories(x, self.categories), X_copy))).to_numpy()
            

    def get_feature_names_out(self, columns):
        return self.out_cols
    
 # Custom transformer to rename columns 


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories: List[List[str]], start_category: int = 0):
        if not isinstance(categories, list):
            raise  ValueError("The categories must be passed as a list os list and columns must be also a list, one list for each columns") 
        if not isinstance(categories[0], list):
            raise  ValueError("The categories must be passed as a list os list, one list for each columns")
        
        if not isinstance(start_category, int):
             raise  ValueError("The starting category must be a integer") 
        
        self.categories = categories
        self.start_category = start_category
        self.encoder = OrdinalEncoder(categories=categories)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X) + self.start_category
    
    def get_feature_names_out(self, columns):
        return self.encoder.get_feature_names_out(columns)
    