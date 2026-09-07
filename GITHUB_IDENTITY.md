# GitHub Identity for Yaoyao project

Use this account for ALL git commits and pushes in D:\TaoVm:

user.name: Tao
user.email: 2584300846@qq.com
repo: https://github.com/TaoLe-si/Yaoyao.git

## Commit
git -C D:\TaoVm commit -m "message"

## Push
git -C D:\TaoVm push origin main
# If non-fast-forward:
git -C D:\TaoVm push --force-with-lease origin main

## Credential
Windows Credential Manager (already configured).


## API server

yaoyao_api.py - FastAPI OpenAI-compatible wrapper around yaoyao_gen.exe
Runs on http://127.0.0.1:11434
Endpoints:
- GET /v1/models
- POST /v1/chat/completions

## Files

- yaoyao_gen.exe - inference-only C++ exe
- yaoyao_v10_seq64.exe - training + inference
- yaoyao_v09_word.cpp - word-level training source
- yaoyao_gen.cpp - inference-only source
- yaoyao_api.py - OpenAI-compatible API server
