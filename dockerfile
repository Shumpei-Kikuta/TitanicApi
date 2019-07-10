# baseimage
FROM python:latest

# 作業directoryの指定
ARG project_dir=/flask_app/
WORKDIR $project_dir

# projectで利用するファイルをコンテナ上へコピー
ADD . $project_dir

RUN apt-get install gcc
RUN pip install --upgrade pip
RUN pip install -r requirement.txt
CMD ["python", "app.py"]