# hello.py
import subprocess

# まずmodel.pyを実行
subprocess.run(["python", "model.py"])

from flask import Flask, request, jsonify
import joblib
import numpy as np

import pandas as pd
from scipy.sparse.linalg import svds

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import logging

app = Flask(__name__)

# モデルの読み込み
model = joblib.load("model.pkl")

# データの読み込み
data = [
    [1, [2, 3]],
    [2, [1, 3]],
    [3, [1, 2, 5]],
    [4, [3, 4]],
    [5, [2, 4, 5]],
    [6, [1, 3, 5]],
    [7, [1, 4]],
    [8, [2, 3, 4]],
    [9, [1, 2]],
    [10, [3, 5]]
]

junre = [
    [1, "中華"],
    [2, "フランス料理"],
    [3, "日本食"],
    [4, "洋食"],
    [5, "イタリアン"]
]




@app.route('/')
def flask_app():
    app.logger.info("flask_app実行")
    return 'おめでとう成功かもあだ!'

@app.route('/predict')
def predict():
    app.logger.info("実行だ!")

    users = [
        [1, [1, 2, 4]],
        [2, [1, 3, 5]],
        [3, [1, 2, 5]],
        [4, [1, 2, 3]]
    ]

    recipes = {
        1: ["中華", "鶏肉"],
        2: ["中華", "豚肉"],
        3: ["洋食", "牛肉"],
        4: ["和食", "魚"],
        5: ["洋食", "鶏肉"]
    }

    recipe_features = pd.DataFrame.from_dict(recipes, orient='index', columns=['Cuisine', 'Main_Ingredient'])
    app.logger.info(recipe_features)

    # 初期化
    user_profiles = []

    # ユーザーごとに特徴量を集計
    for user_id, liked_recipes in users:
        user_data = recipe_features.loc[liked_recipes]
        user_profile = user_data['Cuisine'].value_counts().to_dict()
        user_profile.update(user_data['Main_Ingredient'].value_counts().to_dict())
        user_profiles.append((user_id, user_profile))
    
    app.logger.info(user_profiles)

    user_ids = [user_id for user_id, profile in user_profiles]
    profiles = [profile for user_id, profile in user_profiles]

    df = pd.DataFrame(profiles)
    df['user_id'] = user_ids
    df.set_index('user_id', inplace=True)

    app.logger.info(df)

    # 結果の表示
    for user_id, profile in user_profiles:
        print(f"User {user_id} Profile: {profile}")
    
    return user_profiles
    #return "predict実行だよ!"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=4000)