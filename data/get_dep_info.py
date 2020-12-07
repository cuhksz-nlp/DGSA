import json
from os import path
import argparse
from tqdm import tqdm
import re
from collections import defaultdict
from corenlp import StanfordCoreNLP

FULL_MODEL = './stanford-corenlp-full-2018-10-05'
punctuation = ['。', '，', '、', '：', '？', '！', '（', '）', '“', '”', '【', '】']
chunk_pos = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'INTJ', 'CONJP', 'LST']

def change(char):
    if "(" in char:
        char = char.replace("(", "-LRB-")
    if ")" in char:
        char = char.replace(")", "-RRB-")
    return char
def read_tsv(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line[:2] == '*#':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = re.split('\\s+', line)
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)
    return sentence_list, label_list

def read_txt(file_path):
    sentence_list = []
    with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        for line in fin:
            sent_string, tag_string = line.strip().split('####')
            words = []
            tags = []
            for tag_item in tag_string.split(' '):
                eles = tag_item.split('=')
                if len(eles) == 1:
                    raise Exception("Invalid samples %s..." % tag_string)
                elif len(eles) == 2:
                    word, tag = eles
                else:
                    word = ''.join((len(eles) - 2) * ['='])
                    tag = eles[-1]
                words.append(word)
                tags.append(tag)
            sentence_list.append([words, tags])
    return sentence_list

def request_features_from_stanford(data_dir, flag):
    all_sentences = read_txt(path.join(data_dir, flag + '.txt'))
    sentences_str = []
    for sentence,tags in all_sentences:
        sentence = [change(i) for i in sentence]
        sentences_str.append([sentence, tags])
    all_data = []
    with StanfordCoreNLP(FULL_MODEL, lang='en') as nlp:
        for sentence,tags in tqdm(sentences_str):
            props = {'timeout': '5000000','annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true' ,  'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
            results=nlp.annotate(' '.join(sentence), properties=props)
            results["tags"] = tags
            results["word"] = sentence
            all_data.append(results)
    assert len(all_data) == len(sentences_str)
    with open(path.join(data_dir, flag + '.stanford.json'), 'w', encoding='utf8') as f:
        for data in all_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


class StanfordFeatureProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_json(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                data.append(json.loads(line))
        return data

    def read_features(self, flag):
        all_data = self.read_json(path.join(self.data_dir, flag + '.stanford.json'))
        all_feature_data = []
        for data in all_data:
            sentence_feature = []
            sentences = data['sentences']
            for sentence in sentences:
                tokens = sentence['tokens']
                for token in tokens:
                    feature_dict = {}
                    feature_dict['word'] = token['originalText']
                    sentence_feature.append(feature_dict)

            for sentence in sentences:
                deparse = sentence['basicDependencies']
                for dep in deparse:
                    dependent_index = dep['dependent'] - 1
                    sentence_feature[dependent_index]['dep'] = dep['dep']
                    sentence_feature[dependent_index]['governed_index'] = dep['governor'] - 1

            all_feature_data.append(sentence_feature)
        return all_feature_data


def get_dep(sentence,direct):
    words = [change_word(i["word"]) for i in sentence]
    deps = [i["dep"] for i in sentence]
    dep_matrix = [[0] * len(words) for _ in range(len(words))]
    dep_text_matrix = [["none"] * len(words) for _ in range(len(words))]
    #dep_text_direct_matrix = [["none"] * len(words) for _ in range(len(words))]
    for i, item in enumerate(sentence):
        governor = item["governed_index"]
        dep_matrix[i][i] = 1
        dep_text_matrix[i][i] = "self_loop"
        #dep_text_direct_matrix[i][i] = "self_loop"
        if governor != -1: # ROOT
            dep_matrix[i][governor] = 1
            dep_matrix[governor][i] = 1
            dep_text_matrix[i][governor] = deps[i] if not direct else deps[i]+"_in"
            dep_text_matrix[governor][i] = deps[i] if not direct else deps[i]+"_out"
            # dep_text_direct_matrix[i][governor] = deps[i]+"_in"
            # dep_text_direct_matrix[governor][i] = deps[i]+"_out"

    ret_list = []
    for word, dep, dep_range, dep_text in zip(words, deps, dep_matrix,dep_text_matrix):
        ret_list.append({"word": word, "dep": dep, "range": dep_range,"dep_text":dep_text})
    return ret_list

def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word


def filter_useful_feature(feature_list, feature_type,direct):
    ret_list = []
    for sentence in feature_list:
        if feature_type == "dep":
            ret_list.append(get_dep(sentence,direct))
        else:
            print("Feature type error: ", feature_type)
    return ret_list


def get_feature2count(train_features, test_features, feature2type, direct):
    feature2count = defaultdict(int)
    for feature in [train_features,test_features]:
        for sent in feature:
            for item in sent:
                pos = item[feature2type]
                if direct:
                    # direct
                    feature_in = pos + "_in"
                    feature_out = pos + "_out"
                    feature2count[feature_in] += 1
                    feature2count[feature_out] += 1
                else:
                    #undirect
                    feature2count[pos] += 1
    return feature2count



def generate_knowledge_api(data_dir, feature_type, flag, direct):
    """
    :param data_dir: 数据集地址
    :param feature_type: 知识的类型：pos，chunk，dep
    :return:
    """
    if feature_type not in ["pos", "chunk", "dep"]:
        raise RuntimeError("feature_type should be in ['pos', 'chunk', 'dep']")
    sfp = StanfordFeatureProcessor(data_dir)
    if 'laptop' in  flag:
        train_feature_data = sfp.read_features(flag="laptop_train")
        test_feature_data = sfp.read_features(flag="laptop_test")

    elif 'restaurant' in flag:
        train_feature_data = sfp.read_features(flag="restaurant_train")
        test_feature_data = sfp.read_features(flag="restaurant_test")

    elif 'twitter' in flag:
        train_feature_data = sfp.read_features(flag="twitter_train")
        test_feature_data = sfp.read_features(flag="twitter_test")
    elif 'test' in flag:
        train_feature_data = sfp.read_features(flag="test")
        test_feature_data = sfp.read_features(flag="test")

    train_feature_data = filter_useful_feature(train_feature_data, feature_type=feature_type,direct = direct)
    test_feature_data = filter_useful_feature(test_feature_data, feature_type=feature_type,direct = direct)

    feature2count = get_feature2count(train_feature_data, test_feature_data, feature_type, direct)

    feature2id = {"none": 0,"self_loop":1}
    id2feature = {0: "none",1:"self_loop"}
    index = 2
    for key in feature2count:
        feature2id[key] = index
        id2feature[index] = key
        index += 1
    return train_feature_data, test_feature_data, feature2count, feature2id, id2feature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    args = parser.parse_args()
    request_features_from_stanford(args.data_dir, "train")
    request_features_from_stanford(args.data_dir, "test")

if __name__ == '__main__':
    main()

