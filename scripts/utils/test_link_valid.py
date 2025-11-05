import os
import re

import requests

from swift.utils import get_logger

logger = get_logger()


def check_link(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False


def extract_links_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        links = re.findall(r'\[.*?\]\((.*?)\)', content)
        return links


def check_links_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                if file in ['支持的模型和数据集.md', 'Supported-models-and-datasets.md']:
                    continue
                file_path = os.path.join(root, file)
                logger.info(f'Checking links in file: {file_path}')
                links = extract_links_from_md(file_path)
                for link in links:
                    if not link.startswith(('http://', 'https://')):
                        path = link.rsplit('#', 1)[0]
                        if path:
                            path = os.path.abspath(os.path.join(root, path))
                            if os.path.exists(path):
                                logger.info(f'✅ Link is valid: {link}')
                            else:
                                logger.info(f'❌ Link is broken: {link}')
                        else:
                            logger.info(f'Skipping non-HTTP link: {link}')
                        continue
                    if check_link(link):
                        logger.info(f'✅ Link is valid: {link}')
                    else:
                        if 'huggingface.co' in link:
                            logger.info(f'Link is broken: {link}')
                        else:
                            logger.info(f'❌ Link is broken: {link}')


if __name__ == '__main__':
    folder_path = './'
    check_links_in_folder(folder_path)
