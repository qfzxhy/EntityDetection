from itertools import chain

import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from AspectDetection import wordclasses_load
from EntityDetection import LoadFile
from AspectDetection import restaurant2015 as rst
from AspectDetection import NameListGenerator as nlg
# nltk.corpus.conll2002.fileids()
train_data_path = './data_model/train_data.txt'
test_data_path = './data_model/test_data.txt'
word_class_map = wordclasses_load.load_classes()
traindata_path = '../restaurant2015/ABSA-15_Restaurants_Train_Final.xml'
load = rst.Load(traindata_path)
name_list1,name_list2 = nlg.get_namelist(load.datas,load.labels)
def word2features(sent, i):
    word = sent[i][0]

    postag = sent[i][1]
    word_class = -1
    if word in word_class_map:
        word_class = word_class_map[word]
    on_name_list1 = False
    on_name_list2 = False
    if word in name_list1:
        on_name_list1 = True
    if word in name_list2:
        on_name_list2 = True
    features = {
        'bias':True,
        'word.lower':word.lower(),

        #'word[-3:]=' + word[-3:],
       # 'word[-2:]=' + word[-2:],
        #'word.isupper=%s' % word.isupper(),
       # 'word.istitle=%s' % word.istitle(),
        #'word.isdigit=%s' % word.isdigit(),
         'postag' : postag,
        # #'postag[:2]=' + postag[:2],
         'word.class' : str(word_class),
         'word.on_namelist1':on_name_list1,
         'word.on_namelist2' : on_name_list2,
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        word_class1 = -1
        if word1 in word_class_map:
            word_class1 = word_class_map[word1]
        on_name_list1_1 = False
        on_name_list2_1 = False
        if word1 in name_list1:
            on_name_list1_1 = True
        if word1 in name_list2:
            on_name_list2_1 = True
        features['-1:word.lower'] = word1.lower()
        features['-1:postag'] = postag1
        features['-1:word.class'] = str(word_class1)
        features['-1:word.on_namelist1'] = on_name_list1_1
        features['-1:word.on_namelist2'] = on_name_list2_1
        # features.extend({
        #     '-1:word.lower' : word1.lower(),
        #     #'-1:word.istitle=%s' % word1.istitle(),
        #    # '-1:word.isupper=%s' % word1.isupper(),
             #'-1:postag' : postag1,
        #     #'-1:postag[:2]=' + postag1[:2],
        #     '-1:word.class' : str(word_class1),
        #     '-1:word.on_namelist1' : str(on_name_list1_1),
        #     '-1:word.on_namelist2' : str(on_name_list2_1),
        # })
    else:
        features['BOS'] = True
        #features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        word_class1 = -1
        if word1 in word_class_map:
            word_class1 = word_class_map[word1]
        on_name_list1_1 = False
        on_name_list2_1 = False
        if word1 in name_list1:
            on_name_list1_1 = True
        if word1 in name_list2:
            on_name_list2_1 = True
        features['+1:word.lower'] = word1.lower()

        features['+1:postag'] = postag1
        features['+1:word.class'] = str(word_class1)
        features['+1:word.on_namelist1'] = on_name_list1_1
        features['+1:word.on_namelist2'] = on_name_list2_1
        # features.extend([
        #     '+1:word.lower=' + word1.lower(),
        #    # '+1:word.istitle=%s' % word1.istitle(),
        #    # '+1:word.isupper=%s' % word1.isupper(),
        #     '+1:postag=' + postag1,
        #    # '+1:postag[:2]=' + postag1[:2],
        #     '+1:word.class=' + str(word_class1),
        #     '+1:word.on_namelist1=' + str(on_name_list1_1),
        #     '+1:word.on_namelist2=' + str(on_name_list2_1),
        # ])
    else:
        features['EOS'] = True
        # features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]

train_sents = LoadFile.load_crf_data(train_data_path)
test_sents = LoadFile.load_crf_data(test_data_path)
X_train = [sent2features(s) for s in train_sents]

y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):

    itemseq =  pycrfsuite.ItemSequence(xseq)

    trainer.append(itemseq, yseq)
trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('conll2002-esp.crfsuite')
print len(trainer.logparser.iterations), trainer.logparser.iterations[-1]

tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')

example_sent = test_sents[0]
print(' '.join(sent2tokens(example_sent)), )

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )
y_pred = [tagger.tag(xseq) for xseq in X_test]
print(bio_classification_report(y_test, y_pred))