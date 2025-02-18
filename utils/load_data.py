import pandas as pd

def load_data(path : str, target_value : str):
    df = pd.read_csv(path)
    return [df.drop(target_value, axis=1), df[target_value].map({"Yes" : 1, "No" : 0})]
