from xml.etree import ElementTree as ET 
import re
import pickle


#filename = r"C:\Users\CJ17\Desktop\semcor.gold.key.txt"
#tree = ET.parse(r"C:\Users\CJ17\Desktop\semcor.data.xml")
tree = ET.parse(r"C:\Users\CJ17\Desktop\semeval2007.data.xml")
filename = r"C:\Users\CJ17\Desktop\semeval2007.gold.key.txt"

fn = open(filename,'r')
lines = fn.readlines()


i = 0
ids = [] 
lemmse_sense = []
for line in lines:
    s = line.split()
    ids.append(s[0])
    k = s[1].split('%')
    b = k[1]
    final_sensetag = k[0]+b[5:7]
    #print(final_sensetag)
    lemmse_sense.append(final_sensetag)
    i += 1

fn.close()



root = tree.getroot()

raw_words = []
lemma_words = []
id_numbers = []
flag = []
for node in root.iter('text'):
    for child in node:
        for sub_child in child:
            tmp = sub_child.text
            if re.match('^[0-9a-zA-Z]',tmp):
                raw_words.append(tmp)
                instances = sub_child.get('id')
                #lemma = sub_child.get('lemma')
                if instances in ids:
                    #print(lemmse_sense[ids.index(instances)])
                    lemma_words.append(lemmse_sense[ids.index(instances)])
                    id_numbers.append(instances)
                    flag.append(1)
                    for i in tmp:
                        if i.isspace():
                            lemma_words.append(lemmse_sense[ids.index(instances)])
                            id_numbers.append(instances)
                            flag.append(0)
                        
                else:
                    lemma_words.append(sub_child.get('lemma'))
                    id_numbers.append(instances)
                    flag.append(0)
        
        raw_words.append(1000)
        lemma_words.append(1000)
        id_numbers.append(1000)
        flag.append(0)

print(len(raw_words))
print(len(lemma_words))
print(len(id_numbers))



raw_data = r"input_x.txt"
sense_data = r"output_y.txt"
'''
with open(raw_data,'w') as fout1:
    for item in raw_words:
        if item == 1000:
            fout1.write('\n')
        else:
            item = item.lower()
            fout1.write(item+' ')
    print('finish writing')

with open(sense_data,'w') as fout2:
    for item in lemma_words:
        if item == 1000:
            fout2.write('\n')
        else:
            item = item.lower()
            fout2.write(item+' ')
    print('finish writing')
'''
line = 0
i = 0
line_index = []
number_index = []
context_ids = []
for x in range(len(id_numbers)):
    
    if id_numbers[x] == 1000:
        i = 0
        line += 1
    elif id_numbers[x] in ids and flag[x] == 1:
        line_index.append(line)
        number_index.append(i)
        i += 1  
    else:
        i += 1 


print("finish")
print(len(line_index))
print(max(number_index))
print(line_index)
print(number_index)
dataset = []
dataset.append(line_index)
dataset.append(number_index)
PATH = "index.txt"
fn = open(PATH,'wb')
pickle.dump(dataset,fn)
fn.close()
print('finish pickle')



