import pandas as pd

class TokensDataStorage:
    def __init__(self, tokenized_data):
        self.df = pd.DataFrame(tokenized_data)


    # Debug methods writes a CSV file for output in our CWD
    def write_csv(self):
        self.df.to_csv('tokens.csv', index=False)