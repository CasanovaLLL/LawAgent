import os

import os
import re

import docx
import pypandoc
import fitz  # PyMuPDF


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


def extract_text_from_doc(file_path):
    return pypandoc.convert_file(file_path, 'plain')


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


def save_text_to_file(text, output_folder, file_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, file_name + '.txt')
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)


def process_file(file_path, output_folder):
    text = extract_and_clean_text(file_path)
    if len(text)==0:
        print(f"数据抽取为空；待检查：{file_path}")
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_text_to_file(text, output_folder, file_name)
# 使用示例：
# 假设你想搜索的文件夹路径是'/path/to/your/folder'
file_path_list = find_documents(r'monopoly_data')

for file_path in file_path_list:
    # print(f"正在抽取:{file_path}")
    process_file(file_path,'txt_data')
    # file_txt=extract_and_clean_text(file_path)
    '''
    现在直接根据第三方库读取 doc、docx、pdf文件中的文本数据，并且进行简单的清洗过滤
    文本提取的问题：结构可能会缺失、联想（北京）有限公司垄断案中止调查决定书.pdf 里面是以图片形式存在
    TODO: 去噪/清洗 -> 分块 -> embedding -> index -> search
    法一：全部交给LLM；需要担心数据泄漏、准确性、格式等问题
    法二：基于规则：需要处理很多极端案例
    '''
    # print(file_txt)
