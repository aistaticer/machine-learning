import joblib # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import numpy as np # type: ignore

# ログを出力
print("Start of the script")

# 仮のデータセットを作成
X_train = np.random.rand(100, 5)  # 100サンプル、5特徴量
y_train = np.random.randint(2, size=100)  # 0または1のラベル

# モデルの訓練
model = RandomForestClassifier()
model.fit(X_train, y_train)

# モデルの保存
joblib.dump(model, './model.pkl')

