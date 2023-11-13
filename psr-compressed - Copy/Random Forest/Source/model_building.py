import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def pre_processing(df, prediction_set: bool=False):
    if prediction_set:
        le = pickle.load(open(os.path.join("../", "Data", "Artifacts", "encoder.pkl"), 'rb'))
        df = df.drop(['rolling_return'], axis=1)
        df["Symbol"] = le.transform(df["Symbol"])
        df.replace({-np.inf: -1_000_000, np.inf: 1_000_000}, inplace = True)
        df.fillna(0, inplace=True)

        return df
    else:
        le = LabelEncoder()
        le.fit(df["Symbol"])
        pickle.dump(le, open(os.path.join("../", "Data", "Artifacts", "encoder.pkl"), 'wb'))

        df["Symbol"] = le.transform(df["Symbol"])
        df.fillna(0, inplace=True)

        X = df.drop(['rolling_return'], axis=1)
        X.replace({-np.inf: -1_000_000, np.inf: 1_000_000}, inplace = True)
        y = df[['rolling_return']]

        return X, y


def build_model(X, y):
    rfc = RandomForestClassifier(random_state = 8)
    model = rfc.fit(X, y)
    pickle.dump(model, open(os.path.join("../", "Data", "Artifacts", "model.pkl"), 'wb'))

    return rfc


def load_model():
    model = pickle.load(open(os.path.join("../", "Data", "Artifacts", "model.pkl"), 'rb'))
    
    return model


def predict(df):
    model = load_model()

    le = pickle.load(open(os.path.join("../", "Data", "Artifacts", "encoder.pkl"), 'rb'))
    probs = model.predict_proba(df)
    pd.DataFrame(index=le.inverse_transform(df['Symbol']), data=probs, columns=['0', '1']).sort_values(by='1', ascending=False).to_csv(os.path.join("Test.csv"))
    print(pd.DataFrame(index=le.inverse_transform(df['Symbol']), data=probs, columns=['0', '1']).sort_values(by='1', ascending=False))