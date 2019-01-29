import csv

def load_data_02(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        lst = list(reader)
    m = len(lst)
    data = []
    for l in lst[1::]:
        ll = l[8].strip().split('_')
        print(ll)
        print(list(map(int, ll)))
        pass
        labels = list(map(int, ll))
        case = []
        case.extend(labels) 
        case.extend(l[9:15])
        data.append(case)
    return data, m


fileName = './Data.csv'
data, m = load_data_02(fileName)