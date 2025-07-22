import os
import json
import pandas as pd
import numpy as np
import ast  # Import ast for literal evaluation
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer



def extract_nested_dict(df):
    result = {}
    current_key = None
    
    for _, row in df.iterrows():
        # Check if the first column has a new category
        if not pd.isna(row[0]):
            # Use only the part before the first hyphen for the main key
            main_key = row[0].split("-")[0]
            current_key = main_key.strip()
            result[current_key] = {}
        # If a category is active, process the second column for its entries
        if current_key and not pd.isna(row[1]):
            split_entry = row[1].split(": ", 1)
            key = split_entry[0]
            value = split_entry[1] if len(split_entry) > 1 else ""
            result[current_key][key] = value
    
    return result


def format_data_dict(data_dict):
    output = []
    keys = []
    for main_key, sub_dict in data_dict.items():
        for key, value in sub_dict.items():
            output.append(f"{key}: {value}")
            keys.append(key)
    return "\n".join(output), ', '.join(keys)


def get_name_from_fragment(text, df):
    conditions = []
    
    if 'Species' in df.columns:
        conditions.append(df['Species'].str.contains(text, case=False, na=False))
    
    if 'CanonicalSpecies' in df.columns:
        conditions.append(df['CanonicalSpecies'].str.contains(text, case=False, na=False))
    
    if not conditions:
        # Neither column exists
        print("⚠️ Neither 'Species' nor 'CanonicalSpecies' found in DataFrame.")
        return pd.DataFrame(columns=df.columns)

    # Combine conditions with OR
    combined_mask = conditions[0]
    for cond in conditions[1:]:
        combined_mask |= cond

    return df[combined_mask]

def get_timestamp(): return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def get_info(data_df, counts_df, label_width=19, number_width=6):
    print(f"{'#genera:':>{label_width}} {len(set(data_df['Genus'].values)):>{number_width}}")
    print(f"{'#canonical species:':>{label_width}} {len(set(data_df['CanonicalSpecies'].values)):>{number_width}}")
    print(f"{'#species:':>{label_width}} {len(set(data_df['Species'].values)):>{number_width}}")
    print(f"{'#causal documents:':>{label_width}} {len(data_df):>{number_width}}")
    print(f"{'#sampled documents:':>{label_width}} {counts_df['n_sampled'].sum():>{number_width}}")
    print(f"{'#documents:':>{label_width}} {counts_df['n_articles'].sum():>{number_width}}")

def print_section(msg):
    print()
    print('='*100)
    print(msg)
    print('-'*100)
    print()

def standardize_spelling(name):
    # Remove double quotes and convert the name to lowercase with only the first letter capitalized
    return name.replace('"','').lower().capitalize()    

def get_name_map(df):
    # Create a dictionary to map each genus to a list of associated species
    name_map = defaultdict(list)
    for name in df['bacteria synonym'].values:
        genus, species = name.split()  # Split each name into genus and species
        name_map[genus].append(species)  # Add the species to the genus in the dictionary
    # Convert each list of species to a sorted list without duplicates
    name_map = {genus: list(sorted(set(name_map[genus]))) for genus in name_map}
    return name_map

class SynonymToSynonymsTransformer(object):
    def __init__(self, handle='bacteria_synonym_lookup.csv'):
        # Load the CSV containing bacteria synonyms
        df = pd.read_csv(handle)
        # Map each genus to its species list using the `get_name_map` function
        self.genus_to_species_map = get_name_map(df)
        # Create pairs of standardized synonym names for mapping
        name_pairs = [(standardize_spelling(name1), standardize_spelling(name2)) for name1, name2 in df.values]
        # Create a dictionary that maps synonyms to a standardized name
        self.synonym_to_standard_map = dict(name_pairs)
        # Create a dictionary to map each standard name to its synonyms
        self.standard_to_synonyms_map = defaultdict(list)
        for key in self.synonym_to_standard_map:
            # Append each synonym to the standard name's list of synonyms
            self.standard_to_synonyms_map[self.synonym_to_standard_map[key]].append(key)
              
    def get_genera(self):
        # Return a list of all genera (genus names) with the first letter capitalized
        return [genus.capitalize() for genus in self.genus_to_species_map.keys()]
    
    def get_species_from_genus(self, genus):
        # Return a sorted list of species names for a given genus, in the form "Genus species"
        return sorted(set(['%s %s' % (genus.capitalize(), species) for species in self.genus_to_species_map[genus.lower()]]))

    def genus_to_species(self, genus_list):
        # Given a list of genera, return all species associated with each genus in a flat list
        return sum([self.get_species_from_genus(genus) for genus in genus_list], [])

    def get_standard_name(self, name):
        # Standardize the spelling of the name and return its corresponding standard name
        name = standardize_spelling(name)
        standard_name = self.synonym_to_standard_map[name]
        return standard_name
    
    def get_synonym_names(self, name):
        # Get the standard name and retrieve all synonyms associated with it
        standard_name = self.get_standard_name(name)
        synonyms_names = self.standard_to_synonyms_map.get(standard_name, 'N/A')
        synonyms_names = tuple(synonyms_names)
        return synonyms_names


def convert_numpy_arrays_to_json(df):
    """
    Converts any columns in the DataFrame containing NumPy arrays to JSON strings.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].apply(lambda x: isinstance(x, np.ndarray)).any():
            df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else x)
    return df_copy

def convert_json_to_numpy_arrays(df):
    """
    Converts any JSON strings or Python list-like strings in the DataFrame back to NumPy arrays.
    Handles lists of strings and numbers differently.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        # Check if the column contains strings that look like lists or actual lists
        if df_copy[col].apply(lambda x: isinstance(x, (str, list)) and (
                (isinstance(x, str) and x.startswith('[') and x.endswith(']')) or isinstance(x, list))).any():
            def safe_literal_eval(x):
                try:
                    if isinstance(x, list):
                        # If it's already a list, convert directly to a NumPy array
                        return np.array(x, dtype=object)
                    elif isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                        # Use ast.literal_eval to parse the Python-like list
                        loaded_data = ast.literal_eval(x)
                        if isinstance(loaded_data, list):
                            if all(isinstance(i, (int, float)) for i in loaded_data):
                                return np.array(loaded_data)
                            elif all(isinstance(i, str) for i in loaded_data):
                                return np.array(loaded_data, dtype=object)
                            else:
                                raise ValueError(f"Unsupported data types in list: {loaded_data}")
                        return x  # Not a list, return as-is
                    else:
                        return x  # Not a list or string
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing value in column '{col}': {x}")
                    raise e

            # Apply the function to the column
            df_copy[col] = df_copy[col].apply(safe_literal_eval)
        else:
            # Handle columns that may already contain lists
            df_copy[col] = df_copy[col].apply(lambda x: np.array(x, dtype=object) if isinstance(x, list) else x)

    return df_copy

def save_dataframe_to_csv(df, df_csv_name, directory_name):
    """
    Saves the DataFrame to CSV after converting NumPy arrays to JSON strings.
    """
    # Convert NumPy arrays to JSON strings
    df_to_save = convert_numpy_arrays_to_json(df)
    
    # Define the path to save the CSV file
    csv_file_path = os.path.join(directory_name, df_csv_name)
    
    # Write DataFrame to CSV in the specified directory
    df_to_save.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved as CSV in {csv_file_path}")

def load_dataframe_from_csv(df_csv_name, directory_name):
    """
    Loads the DataFrame from a CSV file and converts JSON strings or Python list-like strings back to NumPy arrays.
    """
    # Define the path to load the CSV file
    csv_file_path = os.path.join(directory_name, df_csv_name)
    
    # Read DataFrame from CSV
    df = pd.read_csv(csv_file_path)
    
    # Convert JSON strings or Python list-like strings back to NumPy arrays
    df_loaded = convert_json_to_numpy_arrays(df)
    
    return df_loaded

def save_results(name, data_df, df_csv_name='data.csv', data_dir=None):
    directory_name = os.path.join(data_dir, f'{name}')
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    save_dataframe_to_csv(data_df, df_csv_name=df_csv_name, directory_name=directory_name)
    
def load_results(name, df_csv_name='data.csv', data_dir=None):
    data_df = pd.DataFrame()
    try:
        directory_name = os.path.join(data_dir, f'{name}')
        data_df = load_dataframe_from_csv(df_csv_name=df_csv_name, directory_name=directory_name)
    except Exception as e:
        print(f"Error during load_results for {name}: {e}")
    return data_df


def add_first_word_column_first(df, column_name, new_column_name):
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    # Create the new column with the first word of each entry in the specified column
    first_word_series = df[column_name].apply(lambda x: x.split()[0] if isinstance(x, str) else "")
    
    # Insert the new column at the first position (index 0)
    df.insert(0, new_column_name, first_word_series)
    
    return df

def move_column(df, column_name, new_position):
    # Remove the column
    col_data = df.pop(column_name)
    # Insert the column at the new position
    df.insert(new_position, column_name, col_data)
    return df

def remove_square_brackets_except_in_columns(df, keyword):
    """
    Removes square brackets '[]' from all entries in DataFrame columns,
    except in columns where the column name contains the user-defined keyword.
    """
    df_copy = df.copy()
    
    # Iterate over each column
    for col in df_copy.columns:
        # If the column name does not contain the keyword, remove square brackets in its entries
        if keyword not in col:
            df_copy[col] = df_copy[col].apply(
                lambda x: remove_brackets(str(x)) if isinstance(x, str) else x
            )
    
    return df_copy

def remove_brackets(text):
    """
    Removes square brackets from the given text.
    """
    return text.replace('[', '').replace(']', '')

def lists_arrays_to_comma_separated_strings(df, keyword):
    """
    Converts lists and NumPy arrays in DataFrame entries to comma-separated strings
    for all columns whose names do not contain the user-defined keyword.
    Strings and other non-iterable objects are left unchanged.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - keyword (str): The keyword to check in column names.
    
    Returns:
    - pd.DataFrame: The transformed DataFrame.
    """
    df_copy = df.copy()
    
    # Iterate over each column
    for col in df_copy.columns:
        # If the column name does not contain the keyword, apply the transformation
        if keyword not in col:
            df_copy[col] = df_copy[col].apply(
                lambda x: ', '.join(map(str, x)) 
                if isinstance(x, (list, np.ndarray)) else x
            )
    
    return df_copy

def get_genus(species): return species.split()[0]

def get_expanded_targets(targets, genera_to_expand):    
    return [species if get_genus(species) in genera_to_expand else get_genus(species) for species in targets]

def add_canonical_name(df):
    mapping_df = pd.read_csv('species_to_canonical_species_mapping.csv')
    mapping_dict = dict(zip(mapping_df['Species'], mapping_df['CanonicalSpecies']))
    df['CanonicalSpecies'] = df['Species'].map(mapping_dict)

    # Reorder columns: move 'CanonicalSpecies' to be the 3rd column
    cols = list(df.columns)
    cols.insert(2, cols.pop(cols.index('CanonicalSpecies')))
    df = df[cols]

    return df


def clean_string(original_string): return ' '.join(original_string.split())

def add_element_count_column_with_index(df: pd.DataFrame, col: str, new_col_name: str, col_index: int = 2):
    # Create the new column with the count of elements in the lists in the specified column
    count_series = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Insert the new column at the specified index
    df.insert(col_index, new_col_name, count_series)
    
    return df

def aggregate_embeddings(df, column_group_by, column_embedding):
    """
    Aggregates embeddings by averaging them for each group and counts the number of instances per group.

    Parameters:
    - df: pandas DataFrame
    - column_group_by: str, name of the column to group by
    - column_embedding: str, name of the column containing numpy arrays (embeddings)

    Returns:
    - result_df: pandas DataFrame with columns [column_group_by, column_embedding, 'counts']
    """
    # Check if the specified columns exist in the DataFrame
    if column_group_by not in df.columns:
        raise ValueError(f"Column '{column_group_by}' not found in DataFrame.")
    if column_embedding not in df.columns:
        raise ValueError(f"Column '{column_embedding}' not found in DataFrame.")

    # Ensure that the embedding column contains numpy arrays
    if not df[column_embedding].apply(lambda x: isinstance(x, np.ndarray)).all():
        raise TypeError(f"All entries in '{column_embedding}' must be numpy arrays.")

    # Define an aggregation function
    def aggregate_group(group):
        embeddings = group[column_embedding]
        stacked = np.stack(embeddings.values)
        mean_embedding = np.mean(stacked, axis=0)
        count = len(embeddings)
        return pd.Series({
            column_embedding: mean_embedding,
            'counts': count
        })

    # Group by the specified column and apply the aggregation function
    aggregated = df.groupby(column_group_by).apply(aggregate_group).reset_index()

    # Ensure columns are in the desired order
    result_df = aggregated[[column_group_by, column_embedding, 'counts']]

    return result_df


def safe_norm_extended(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        arr = np.array(x)
        if arr.size == 0:
            return np.nan  # Handle empty arrays
        return np.linalg.norm(arr)
    else:
        return np.nan  # or any default value you prefer
    

def preprocess_list_column(entry):
    """
    Pre-processes an entry in the specified column.
    
    - If the entry is a list, returns it as is.
    - If the entry is a string, splits it by commas and strips whitespace.
    - Otherwise (e.g., NaN), returns an empty list.
    
    Parameters:
    - entry: The entry to preprocess.
    
    Returns:
    - list of strings
    """
    if isinstance(entry, list):
        return entry
    elif isinstance(entry, str):
        return [item.strip() for item in entry.split(',') if item.strip()]
    else:
        return []

def add_one_hot_encoding(df, col, unique_classes, onehot_column_name='onehot_embedding'):
    """
    Adds one-hot encoded features for a specified column in the DataFrame using only specified classes.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - col (str): The column name containing lists of strings.
    - unique_classes (list): List of unique classes to use for one-hot encoding.
    - onehot_column_name (str): The name of the new column to store one-hot encoded arrays.

    Returns:
    - df (pd.DataFrame): The original DataFrame with the new one-hot encoded column added.
    """
    # Preprocess the target column
    summary_series = df[col].apply(preprocess_list_column)
    
    # Initialize MultiLabelBinarizer with the provided classes
    mlb = MultiLabelBinarizer(classes=unique_classes)
    
    # Fit and transform the data
    one_hot = mlb.fit_transform(summary_series)
    
    # Ensure one-hot encoding columns align with `unique_classes`
    one_hot_df = pd.DataFrame(one_hot, columns=unique_classes, index=df.index)
    
    # Assign the one-hot encoded DataFrame as a new column
    df[onehot_column_name] = one_hot_df.values.tolist()
    
    return df

