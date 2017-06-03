from xml.dom.minidom import parse
from nlp_tools.stanford_parser import parser

#from nltk.tokenize import word_tokenize
import nltk
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import codecs
def parse_xml(file_path):
    datas = []
    labels = []
    DOMTree = parse(file_path)
    Data = DOMTree.documentElement
    sentence_list = Data.getElementsByTagName("sentence")
    for sentence in sentence_list:
        text_node = sentence.getElementsByTagName("text")[0]
        text_str = text_node.childNodes[0].data
        datas.append(text_str)
        opinions = sentence.getElementsByTagName("Opinions")
        label = []
        if len(opinions) > 0:
            opinions = opinions[0].getElementsByTagName("Opinion")
            for opinion in opinions:
                dic = {}
                dic['target'] = opinion.getAttribute('target')
                dic['category'] = opinion.getAttribute('category')
                dic['polarity'] = opinion.getAttribute('polarity')
                dic['from'] = opinion.getAttribute('from')
                dic['to'] = opinion.getAttribute('to')
                label.append(dic)
        labels.append(label)
    return datas,labels

#generate crf formal data,and write to txt
#formal:[(u'Melbourne', u'NP', u'B-LOC'),u'(', u'Fpa', u'O'),]
def get_tokens(parse_result,sent):

    tokens = []
    for x in parse_result.tokens:
        r = list(x)
        tokens.append(sent[int(r[0]):int(r[1])])
    return tokens
def get_pos_tags(parse_result):
    return list(parse_result.posTags)

def data_crf_generate(file_path,crf_file_path):
    stanford_parser = parser.Parser()
    sents,sents_labels = parse_xml(file_path)
    train_datas = []
    for sent,sent_labels in zip(sents,sents_labels):
        parse_result = stanford_parser.parseToStanfordDependencies(sent)
        tokens = get_tokens(parse_result,sent)
        pos_tags = get_pos_tags(parse_result)
        train_data = []
        for i,token in enumerate(tokens):
            train_data.append((tokens[i],pos_tags[i],'O'))
        for sent_label in sent_labels:
            if sent_label['target'] != 'NULL':
                process_label(sent_label,sent,train_data)
        train_datas.append(train_data)

    writer = codecs.open(crf_file_path,'w','utf-8')
    for train_data in train_datas:
        writer.write(str(train_data)+"\n")
    writer.flush()
    writer.close()

#middle
def process_label(sent_label,sent,train_data):

    begin = int(sent_label['from'])
    end = int(sent_label['to'])
    #没有空格
    new_begin = len(sent[:begin].replace(' ',''))
    new_end = len(sent[:end].replace(' ',''))
    curid = 0
    flag = False

    for i,data_tuple in enumerate(train_data):
        if curid == new_begin:
            train_data[i] = (train_data[i][0],train_data[i][1],'B-TERM')
        if curid > new_begin and curid < new_end and i > 0 and 'TERM' in train_data[i-1][2]:
            train_data[i] = (train_data[i][0],train_data[i][1],'I-TERM')
        if  curid + len(data_tuple[0]) == new_end:
            flag = True
            break
        curid += len(data_tuple[0])
    if not flag:
        print(sent)
        print("error")



#error info
# Food-awesome.
# error
# Although the tables may be closely situated, the candle-light, food-quality and service overcompensate.
# error
# I really like both the scallops and the mahi mahi (on saffron risotto-yum!).
# error
# You can get a completely delish martini in a glass (that's about 2 1/2 drinks) for $8.50 (I recommend the Vanilla Shanty, mmmm!) in a great homey setting with great music.
# error
# You can get a completely delish martini in a glass (that's about 2 1/2 drinks) for $8.50 (I recommend the Vanilla Shanty, mmmm!) in a great homey setting with great music.
# error
# You can get a completely delish martini in a glass (that's about 2 1/2 drinks) for $8.50 (I recommend the Vanilla Shanty, mmmm!) in a great homey setting with great music.
# error



if __name__ == '__main__':
    testdata_path = '../restaurant2015/ABSA15_Restaurants_Test.xml'
    traindata_path = '../restaurant2015/ABSA-15_Restaurants_Train_Final.xml'
    data_crf_generate(traindata_path,'./data_model/train_data.txt')
    data_crf_generate(testdata_path,'./data_model/test_data.txt')
    # sent_label = {'from':6,'to':14}
    # sent = 'great hot dogs..'
    #
    # train_data = [('great', 'JJ', 'O'), ('hot', 'JJ', 'O'), ('dogs', 'NN', 'O'),('.','.','O'),('.','.','O')]
    # process_label(sent_label,sent,train_data)

