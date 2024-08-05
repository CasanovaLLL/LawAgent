import time

import pandas as pd
from DrissionPage import SessionPage
from DrissionPage import WebPage, ChromiumOptions
from DrissionPage import SessionPage, SessionOptions

need_download_path =r'【需下载】反垄断行政处罚_20210101之后_需要通过外部链接下载.xlsx'

need_download_df = pd.read_excel(need_download_path)

print(need_download_df.columns)

co = ChromiumOptions().set_paths(local_port=9111, user_data_path=r'chrom_data\test1')
# co = co.headless()
page = WebPage(chromium_options=co)
for idx,row in need_download_df.iterrows():
    download_url=row['外部链接']
    litigant=row['当事人']
    print(f"正在下载:{download_url}中的数据...")
    page.get(url=download_url)
    download_url_list = page.eles("tag:a")
    for use_url_ele in download_url_list:
        use_url = use_url_ele.link
        if use_url is not None:
            if '.doc' in use_url or '.pdf' in use_url:
                time.sleep(2)
                file_name = use_url_ele.text
                use_url_ele.click.to_download(save_path='tmp', rename=f'{litigant}_{file_name}')
                page.wait.download_begin()
                page.wait.all_downloads_done()
                print(f"下载链接：{use_url} -> {file_name} 下载完成")
            # res = page.download(use_url, 'test')
            # print(res)
    # print("xxx")

# print("xxx")