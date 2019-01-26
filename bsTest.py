from bs4 import BeautifulSoup                                                   
import os

def rename_files(filename):
     file_to_open = os.path.expanduser(filename)
     f = open(file_to_open)
     print(f.read())

def match_class(target):                                                        
    def do_match(tag):                                                          
        classes = tag.get('class', [])                                          
        return all(c in classes for c in target)                                
    return do_match   

filename = '~/Downloads/01_OS_3.htm'
file_to_open = os.path.expanduser(filename)


html = open(file_to_open)
soup = BeautifulSoup(html, 'html.parser')                                                    
print(soup.find_all(match_class(["txt-body"])))