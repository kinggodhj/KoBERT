def read_file(path):
  text=[]
  f = open('./tokenized/tokenized/'+path, 'r', encoding='utf-8')
  lines = f.readlines()
  for line in lines:
      text.append(line.rstrip())
  f.close()
  return text

def write_file(path, data):
  f = open(path, 'w', encoding='utf-8')
  for i in range(len(data)):
    f.write(data[i])
    f.write('\n')
  f.close()

data=[]
#file=['ratings_komoran.txt', 'wiki_ko_mecab.txt', 'corpus_mecab_jamo.txt']
file=['wiki_ko_mecab.txt']

for f in file:
  text=read_file(f)
  #data.extend(text)
  sentences = ["[CLS] " + query + " [SEP]" for query in text]
  write_file('bert_'+f, sentences)

