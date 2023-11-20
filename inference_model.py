from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import pickle

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Функция принимает объект класса Item, преобразует его в соответствии с логикой построения модели
    и делает предсказание для 1 объекта
    """

    test_item = dict(item)

    df = pd.DataFrame.from_dict([test_item])
    df = df.drop(columns=['name', 'torque'])

    df['mileage'] = df['mileage'].apply(lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)
    df['engine'] = df['engine'].apply(lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)
    df['max_power'] = df['max_power'].apply(lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)

    df[['seats', 'engine']] = df[['seats', 'engine']].astype(int)

    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)

    #y_test = df['selling_price']
    X_test = df.drop(columns=['selling_price', 'fuel', 'seller_type', 'transmission', 'owner'])

    X_test = pd.DataFrame(data['poly'].transform(X_test))
    X_test_scaled = pd.DataFrame(data['scaler'].transform(X_test), columns=X_test.columns)

    cat_lst = ['fuel', 'seller_type', 'transmission', 'owner']
    X_test_cat = df.drop(columns=['selling_price'])
    X_test_cat = X_test_cat[cat_lst]
    X_test_cat_ohe = pd.DataFrame(data['ohe'].transform(X_test_cat).toarray())

    X_test_num_with_cat = pd.concat([X_test_scaled, X_test_cat_ohe], axis=1)

    X_test_num_with_cat.columns = X_test_num_with_cat.columns.astype(str)

    return data['best_model'].predict(X_test_num_with_cat)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    """
        Функция принимает список объектов класса Item, преобразует их в соответствии с логикой построения модели
        и делает список предсказаний цены для признакового описания каждого объекта
    """

    preds = []
    for obj in items:
        test_item = dict(obj)

        df = pd.DataFrame.from_dict([test_item])
        df = df.drop(columns=['name', 'torque'])

        df['mileage'] = df['mileage'].apply(lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)
        df['engine'] = df['engine'].apply(lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)
        df['max_power'] = df['max_power'].apply(
            lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)

        df[['seats', 'engine']] = df[['seats', 'engine']].astype(int)

        with open('data.pickle', 'rb') as f:
            data = pickle.load(f)

        #y_test = df['selling_price']
        X_test = df.drop(columns=['selling_price', 'fuel', 'seller_type', 'transmission', 'owner'])

        X_test = pd.DataFrame(data['poly'].transform(X_test))
        X_test_scaled = pd.DataFrame(data['scaler'].transform(X_test), columns=X_test.columns)

        cat_lst = ['fuel', 'seller_type', 'transmission', 'owner']
        X_test_cat = df.drop(columns=['selling_price'])
        X_test_cat = X_test_cat[cat_lst]
        X_test_cat_ohe = pd.DataFrame(data['ohe'].transform(X_test_cat).toarray())

        X_test_num_with_cat = pd.concat([X_test_scaled, X_test_cat_ohe], axis=1)

        X_test_num_with_cat.columns = X_test_num_with_cat.columns.astype(str)
        preds.append(data['best_model'].predict(X_test_num_with_cat))
    return preds



