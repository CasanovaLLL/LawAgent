from setuptools import setup, find_packages

# 读取README.md文件内容作为long_description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
# 读取requirements.txt文件内容
with open("requirements.txt", "r", encoding="utf-8") as requirements_file:
    requirements = requirements_file.read().splitlines()

# 过滤掉requirements.txt中可能的注释和空行
requirements = [req for req in requirements if req and not req.startswith("#")]

# 设置项目信息
setup(
    name="LawAgent",
    version="0.0.9",
    author="Lyx",
    description="A Law Agent for Antimonopoly Law",
    long_description=long_description,
    long_description_content_type="text/markdown",  # 指定长描述内容类型，如果使用Markdown
    url="https://github.com/CasanovaLLL/LawAgent",  # 仓库地址
    packages=find_packages(exclude=["data/", "venv/"]),  # 自动发现并包含所有包
    python_requires='>=3.10',  # 指定Python兼容版本
    entry_points={
        "console_scripts": [
            "start-dify-server=LawAgent.ToolServer.server:main",
            'document-write=LawAgent.gui.server:start_web_ui'
        ]
    },
    install_requires=requirements,
    extras_require={
    },
)
