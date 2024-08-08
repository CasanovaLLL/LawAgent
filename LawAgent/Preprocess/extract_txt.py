import os

import os
import re

import docx
import pypandoc
import fitz  # PyMuPDF
import win32com.client
# # 下载 pandoc 并设置 pypandoc 使用该版本
# pypandoc.download_pandoc()
def find_documents(root_dir):
    """
    递归查找指定文件夹下的所有.doc、.docx和.pdf文件，并返回它们的绝对路径列表。

    :param root_dir: 要搜索的文件夹的路径
    :return: 包含找到的文件绝对路径的列表
    """
    file_types = ('.doc', '.docx', '.pdf')  # 定义要查找的文件后缀
    found_files = []  # 初始化一个空列表来存储找到的文件路径

    # os.walk遍历文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件后缀是否是我们要找的类型
            if filename.endswith(file_types):
                # 构造文件的绝对路径
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                # 将绝对路径添加到列表中
                found_files.append(full_path)

    return found_files


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)


# def extract_text_from_doc(file_path):
#     return pypandoc.convert_file(file_path, 'plain')

def extract_text_from_doc(file_path):
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(file_path)
    text = doc.Range().Text
    doc.Close()
    word.Quit()
    return text

def extract_text_from_pdf(file_path):
    text = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text.append(page.get_text())
    return '\n'.join(text)


def clean_text(text):
    # 移除特殊字符和非中文字符，仅保留中文字符、标点符号和数字
    text = re.sub(r'[^\u4e00-\u9fa5，。！？；、：0-9a-zA-Z\s\n]', '', text)
    return text


def extract_and_clean_text(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.docx':
        text = extract_text_from_docx(file_path)
    elif file_extension == '.doc':
        text = extract_text_from_doc(file_path)
    elif file_extension == '.pdf':
        text = extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file type: {}".format(file_extension))

    return clean_text(text)


def save_text_to_file(text, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

def process_file(file_path, input_folder, output_folder):

    text = extract_and_clean_text(file_path)
    relative_path = os.path.relpath(file_path, input_folder)
    output_path = os.path.join(output_folder, os.path.splitext(relative_path)[0] + '.txt')
    save_text_to_file(text, output_path)

def process_folder(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.docx', '.doc', '.pdf')):
                file_path = os.path.join(root, file)
                try:
                    print(f"正在处理：{file_path}")
                    process_file(file_path, input_folder, output_folder)
                except BaseException as e:
                    print(f"Debug :{str(e)}")



input_folder=r'data/monopoly_data'
output_folder=r'data/monopoly_txt_data'
process_folder(input_folder, output_folder)