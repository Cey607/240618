@echo off

REM 启动Python虚拟环境（如果有的话）
REM activate your_python_virtual_env

REM 设置Flask应用的入口文件
set FLASK_APP=launcher.py

REM 启动Flask应用
start cmd /k flask run

REM 打开浏览器并访问网页
start "" "http://localhost:5000/"
