FROM python:3.8

# 作業ディレクトリを/appに設定
WORKDIR /app

COPY . /app

# requirements.txtで指定された必要なパッケージをインストール
RUN pip install -r requirements.txt

# ポートの公開
EXPOSE 5000

# コンテナ起動時にhello.pyを実行
CMD ["python", "hello.py"]