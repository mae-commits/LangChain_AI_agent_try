FROM python:3.10
USER root

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip

# requirements.txtがある場合、そのファイルをコンテナ内にコピーしてからインストール
COPY requirements.txt /workspace/requirements.txt

# cudfのインストールを無効にして依存関係をインストール
RUN sed -i '/cudf/d' /workspace/requirements.txt && \
    pip install -r /workspace/requirements.txt

# JupyterとIPythonカーネルをインストール
RUN pip install ipykernel jupyter