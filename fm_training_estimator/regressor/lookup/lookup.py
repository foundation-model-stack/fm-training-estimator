# Third Party
import pandas


class LookupRegressor:
    def __init__(self):
        self.data = None

    def load(self, data_path):
        self.data = pandas.read_csv(data_path)

    def run(self, X: dict):
        query = ""
        for key, val in X.items():
            if isinstance(val, str):
                query += f' and {key} == "{val}"'
            else:
                query += f" and {key} == {val}"
        query = query[5:]

        res = self.data.query(query)
        res = res.drop(columns=X.keys())

        return res
