#!/usr/bin/env bash

source ./venv/bin/activate
pipreqs  --force --mode no-pin  --savepath ./requirements.txt ./LawAgent