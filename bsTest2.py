import os
from bs4 import BeautifulSoup
# import requests


# page = requests.get('https://web.archive.org/web/20121007172955/https://www.nga.gov/collection/anZ1.htm')
# soup = BeautifulSoup(page.text, 'html.parser')
filename = '~/Downloads/01_OS_3.htm'
file_to_open = os.path.expanduser(filename)

html = open(file_to_open)
soup = BeautifulSoup(html, 'html.parser') 
# print(soup.get_text())


text_body_list = soup.find_all(class_='txt-body')
print(type(text_body_list))
print(text_body_list[0])

for t in text_body_list:
    print(''.join(t.findAll(text=True)))

judg1 = soup.find_all(class_='Judg-1')
print(type(judg1))
print(len(judg1))


for j in judg1[164::]:
    print(''.join(j.findAll(text=True)))

