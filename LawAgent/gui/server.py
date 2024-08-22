import os
import argparse
import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from LawAgent.gui.gui import MyWebUI
from LawAgent.Tools.file_operator import work_dir
from LawAgent.Agents.document_generate_team import build_agent

app = FastAPI()


@app.get("/file")
async def download_file(name: str):
    if not name.startswith(work_dir) or not os.path.isfile(name):
        return HTTPException(status_code=400, detail="文件路径错误")
    return FileResponse(name)


gui = MyWebUI.run(build_agent)
app = gr.mount_gradio_app(app, gui, path="/")


def start_web_ui():
    # 创建解析器
    parser = argparse.ArgumentParser(description='Start a Gradio web UI.')

    # 添加端口号参数
    parser.add_argument('--port', type=int, default=10193, help='Port number to run the server on.')

    # 解析命令行参数
    args = parser.parse_args()

    # 获取端口号
    port = args.port
    uvicorn.run(app, host="0.0.0.0", port=port)
