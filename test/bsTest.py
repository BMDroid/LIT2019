import os
from bs4 import BeautifulSoup
# import requests


# page = requests.get('https://web.archive.org/web/20121007172955/https://www.nga.gov/collection/anZ1.htm')
# soup = BeautifulSoup(page.text, 'html.parser')
filename = '~/workspace/github/lit2019/01_OS_3.htm'
file_to_open = os.path.expanduser(filename)

html = open(file_to_open)
soup = BeautifulSoup(html, 'html.parser') 
# print(soup.get_text())


# ------ Get the Acts ------

text_body_list = soup.find_all(class_='txt-body')
print(text_body_list[0])
# print(type(text_body_list))

for t in text_body_list:
    print(''.join(t.findAll(text=True)))

# ------ Get the Judge 1 ------
judg1 = soup.find_all(class_='Judg-1')
print(len(judg1))
# print(type(judg1))

for j in judg1[160::]:
    print(''.join(j.findAll(text=True)))

# ------- Try to find the Conlution ------
heading = soup.find_all(class_='Judg-Heading-1')
print(len(heading))

for h in heading:
    if str(h.findAll(text=True)[0]) == 'Conclusion':
        print('Conclution Found')
    print(''.join(h.findAll(text=True)))

# ------ function return the matching class text in different lines ------

def find_matching_classes(className):
    text = ''
    lst = soup.find_all(class_=className)
    for l in lst:
        text += l.findAll(text=True)[0] + '\n'
    return text

className = 'Judg-Heading-2'
print(find_matching_classes(className))


