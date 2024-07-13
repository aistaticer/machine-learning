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

    logging.basicConfig(level=logging.INFO)
    app_logger = logging.getLogger(__name__)

    # ユーザーIDとレシピIDをDataFrameに変換
    df = pd.DataFrame(data, columns=['user_id', 'recipe_ids'])

    # レシピIDとジャンルをDataFrameに変換
    junre_df = pd.DataFrame(junre, columns=['recipe_id', 'genre'])

    # マルチラベルバイナライザーを使用して、ユーザーの好みをジャンルで表現
    mlb = MultiLabelBinarizer(classes=junre_df['recipe_id'])
    X = mlb.fit_transform(df['recipe_ids'])

    # ターゲットとして、ユーザーが選んだ全ジャンルのラベルを使う
    y = mlb.transform(df['recipe_ids'])

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ランダムフォレスト分類器を使用
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # テストデータに対して予測
    predictions = model.predict(X_test)
    app_logger.info("テスト")
    app_logger.info(predictions)

    # テストデータの予測結果をログに出力
    for i, (true, pred) in enumerate(zip(y_test, predictions)):
        app_logger.info(f"True genres: {mlb.inverse_transform(np.array([true]))}, Predicted genres: {mlb.inverse_transform(np.array([pred]))}")

    # 新しい仮のデータセットを作成
    X_new = np.random.rand(10, 5)  # 10サンプル、5特徴量

    # 予測を行う
    predictions = model.predict(X_new)

    # 予測結果を出力
    #app.logger.info("Predictions: %s", predictions)
    
    return "predict実行だ!"
    #data = request.get_json(force=True)
    #prediction = model.predict(np.array(data['features']).reshape(1, -1))
    #print(prediction)
    #print(jsonify({'prediction': int(prediction[0])}))
    #return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=4000)