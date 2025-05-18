import os
import re
from urllib.parse import urlparse

import requests


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
                file_path = os.path.join(root, file)
                print(f'Checking links in file: {file_path}')
                links = extract_links_from_md(file_path)
                for link in links:
                    if not link.startswith(('http://', 'https://')):
                        path = link.rsplit('#', 1)[0]
                        if path:
                            path = os.path.abspath(os.path.join(root, path))
                            if os.path.exists(path):
                                print(f'✅ Link is valid: {link}')
                            else:
                                print(f'❌ Link is broken: {link}')
                        else:
                            print(f'Skipping non-HTTP link: {link}')
                        continue
                    if check_link(link):
                        print(f'✅ Link is valid: {link}')
                    else:
                        print(f'❌ Link is broken: {link}')


if __name__ == '__main__':
    folder_path = './'
    check_links_in_folder(folder_path)
