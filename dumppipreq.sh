#!/usr/bin/env bash

source ./venv/bin/activate
pipreqs --pypi-server https://mirrors.ustc.edu.cn/pypi/web/simple \
  --force --mode compat \
  --savepath ./requirements.txt ./LawAgent