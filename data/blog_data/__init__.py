import pandas as pd
import tempfile
GITHUB_LIMIT = 99*1000*1000
def load_data_for_classification(filename = 'vocab_indexed_blog_data.csv', chunkSize = GITHUB_LIMIT):
    #if info file available read the split details
    with open('info.txt') as info_file:
        info = info_file.readline().split(',')
        name = info[0]
        nChunks = int(info[2])
        chunkSize = int(info[3])

    tf = tempfile.NamedTemporaryFile(delete=False)
    joinFiles(name, nChunks, chunkSize, tf.name)
    df = pd.read_csv(tf.name, encoding = 'utf-8')
    tf.close()
    return df



# define the function to split the file into smaller chunks
def splitFile(inputFile,chunkSize = GITHUB_LIMIT):
    #read the contents of the file
    f = open(inputFile, 'rb')
    data = f.read()
    f.close()

# get the length of data, ie size of the input file in bytes
    bytes = len(data)

#calculate the number of chunks to be created
    noOfChunks= bytes/chunkSize
    if(bytes%chunkSize):
        noOfChunks+=1

#create a info.txt file for writing metadata
    f = open('info.txt', 'w')
    f.write(inputFile+','+'chunk,'+str(noOfChunks)+','+str(chunkSize))
    f.close()

    chunkNames = []
    for idx,i in enumerate(range(0, bytes+1, chunkSize)):
        fn1 = inputFile + "_%s" % idx
        chunkNames.append(fn1)
        f = open(fn1, 'wb')
        f.write(data[i:i+ chunkSize])
        f.close()

#define the function to join the chunks of files into a single file
def joinFiles(fileName, noOfChunks, chunkSize, outputFile):
    dataList = []
    for i in range(noOfChunks):
        chunkNum=i * chunkSize
        chunkName = fileName+'_%s'%i
        f = open(chunkName, 'rb')
        dataList.append(f.read())
        f.close()
    f2 = open(outputFile, 'wb')
    for data in dataList:
        f2.write(data)
    f2.close()




def load_data():
    df = pd.read_csv('blog_data.csv', encoding = 'utf-8')
    return df

import pickle

def load_vocab():
    vocab = {}
    with open('vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    rev_vocab = { vocab[key]:key for key in vocab }
    return vocab, rev_vocab

def replace_missing_words_with_unk():
    df = load_data()
    unk = 0
    valid = 0
    vocab,rev_vocab = load_vocab()
    column = []
    for index, row in df.iterrows():
        sent_in_numbers =  []
        for word in row['sentence'].split(' '):
            if(word in vocab):
                sent_in_numbers.append(vocab[word])
                valid+=1
            else:
                sent_in_numbers.append(vocab['<UNK>'])
                unk+=1

        column.append(sent_in_numbers)
    df['sent_in_numbers'] = column

    print unk,valid
    return df



def dump_blogs_into_csv():
    '''
        import xml.etree.ElementTree as ET
        import glob
        import spacy

        nlp = spacy.load('en')
        filenames = glob.glob('*.xml')
        count = len(filenames)
        sentence_list = []
        for idx,filename in enumerate(filenames):
            tags = filename.split('.')[:-1]
            post_id = tags[0]
            gender = tags[1] 
            age = tags[2]
            industry = tags[3] 
            astrological_sign = tags[4]

            if age >=13 and age <=17:
                age = 10
            elif age>=23 and age <=27:
                age = 20
            elif age>=33 and age <=47:
                age = 30

            if idx > 5:
                break
            print idx
            try:
                tree = ET.iterparse(filename)
            except:
                continue

            while(1):
                try:
                    _, node = next(tree)
                except:
                    break
                #print sentence_list
                if node.tag == 'post':
                    post = nlp(unicode(node.text.strip().lower()))
                    for sent in list(post.sents):
                        res = []
                        for lex in sent:
                            if lex.like_num:
                                res.append('<#>')
                            elif not lex.is_space:
                                res.append(lex.orth_)
                        if len(sent) > 5 and len(sent) < 30:
                            sentence_list.append(pd.DataFrame([[post_id, gender, age, industry, astrological_sign, ' '.join(res)]]))
                        



        #Concatinate the sentence list into one DataFrame
        df = pd.concat(sentence_list)
        df = df.reset_index(drop = True)
        #defien the columns
        df.columns = ['post_id', 'gender', 'age', 'industry', 'astrological_sign', 'sentence']
        
        df.to_csv('blog_data.csv', encoding = 'utf-8')
    '''
    return

