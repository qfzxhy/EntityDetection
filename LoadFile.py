import codecs
def load_crf_data(file_path):
    reader = codecs.open(file_path,'r','utf-8')
    datas = []
    for line in reader.readlines():
        datas.append(eval(line))
    reader.close()
    return datas

if __name__ =='__main__':
    load_crf_data('./data_model/test_data.txt')