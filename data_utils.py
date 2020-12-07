from __future__ import absolute_import, division, print_function

import logging
import os
import json
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from BERT import WEIGHTS_NAME,CONFIG_NAME

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, dep=None, adj=None, dep_text=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

        self.dep = dep
        self.adj = adj
        self.dep_text = dep_text

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 valid_ids=None, label_mask=None, b_use_valid_filter=False,
                 adj_matrix=None, dep_matrix=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

        self.b_use_valid_filter = b_use_valid_filter

        self.adj_matrix = adj_matrix
        self.dep_matrix = dep_matrix

def tag2ts(ts_tag_sequence):
    """
    transform ts tag sequence to targeted sentiment
    :param ts_tag_sequence: tag sequence for ts task
    :return:
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        # current position and sentiment
        # tag O and tag EQ will not be counted
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        if sentiment != 'O':
            # current word is a subjective word
            sentiments.append(sentiment)
        if pos == 'S':
            # singleton
            ts_sequence.append((i, i, sentiment))
            sentiments = []
        elif pos == 'B':
            beg = i
            if len(sentiments) > 1:
                # remove the effect of the noisy I-{POS,NEG,NEU}
                sentiments = [sentiments[-1]]
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1
    return ts_sequence

def ot2bieos_ts(ts_tag_sequence):
    """
    ot2bieos function for targeted-sentiment task, ts refers to targeted -sentiment / aspect-based sentiment
    :param ts_tag_sequence: tag sequence for targeted sentiment
    :return:
    """
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    prev_pos = '$$$'
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O' or cur_ts_tag == 'EQ':
            # when meet the EQ label, regard it as O label
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            # cur_pos is T
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_ts_sequence.append('S-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('B-%s' % cur_sentiment)
            else:
                # prev_pos is T and new_cur_pos can only be I or E
                if i == n_tags - 1:
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence

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
        all_data = self.read_json(os.path.join(self.data_dir, flag + '.stanford.json'))
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


            tags = data["tags"]
            for i,tag in enumerate(tags):
                sentence_feature[i]['tag'] = tag

            all_feature_data.append(sentence_feature)
        return all_feature_data

def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word

def get_dep(sentence,direct):
    words = [change_word(i["word"]) for i in sentence]
    tags = [i["tag"] for i in sentence]
    deps = [i["dep"] for i in sentence]
    dep_matrix = [[0] * len(words) for _ in range(len(words))]
    dep_text_matrix = [["none"] * len(words) for _ in range(len(words))]
    for i, item in enumerate(sentence):
        governor = item["governed_index"]
        dep_matrix[i][i] = 1
        dep_text_matrix[i][i] = "self_loop"
        if governor != -1: # ROOT
            dep_matrix[i][governor] = 1
            dep_matrix[governor][i] = 1
            dep_text_matrix[i][governor] = deps[i] if not direct else deps[i]+"_in"
            dep_text_matrix[governor][i] = deps[i] if not direct else deps[i]+"_out"

    ret_list = []
    for word, tag, dep, dep_range, dep_text in zip(words, tags, deps, dep_matrix,dep_text_matrix):
        ret_list.append({"word": word, "tag":tag, "dep": dep, "adj": dep_range,"dep_text":dep_text})
    return ret_list

def filter_useful_feature(feature_list, feature_type, direct):
    ret_list = []
    for sentence in feature_list:
        if feature_type == "dep":
            ret_list.append(get_dep(sentence, direct))
        else:
            print("Feature type error: ", feature_type)
    return ret_list


class E2EASAOTProcessor(object):
    def __init__(self, direct=False, dev=False):
        self.direct = direct
        self.train_examples = None
        self.dev_examples = None
        self.test_examples = None
        self.feature2id = {"none": 0, "self_loop": 1}
        self.dev = dev

    def get_type_num(self):
        type_num = 100 if self.direct else 50
        return type_num

    def get_label_num(self):
        label_list = self.get_labels()
        return len(label_list) + 1

    @classmethod
    def _read_tsv(cls, input_file):
        '''
        read file
        return format :
        '''
        f = open(input_file)
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.strip().split('\t')
            sentence.append(splits[0])
            label.append(splits[-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data

    def load_data(self, data_dir):
        if self.train_examples is None:
            self.train_examples = self._create_examples(
                self.get_knowledge_feature(data_dir,flag="train"), "train")
        if self.dev_examples is None and self.dev is True:
            self.dev_examples = self._create_examples(
                self.get_knowledge_feature(data_dir,flag="dev"), "dev")
        if self.test_examples is None:
            self.test_examples = self._create_examples(
                self.get_knowledge_feature(data_dir,flag="test"), "test")

    def get_train_examples(self, data_dir):
        """See base class."""
        self.load_data(data_dir)
        if self.train_examples is None:
            self.train_examples = self._create_examples(
                self.get_knowledge_feature(data_dir,flag="train"), "train")
        return self.train_examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        self.load_data(data_dir)
        if self.dev_examples is None:
            self.dev_examples = self._create_examples(
                self.get_knowledge_feature(data_dir,flag="dev"), "dev")
        return self.dev_examples

    def get_test_examples(self, data_dir):
        """See base class."""
        self.load_data(data_dir)
        if self.test_examples is None:
            self.test_examples = self._create_examples(
                self.get_knowledge_feature(data_dir,flag="test"), "test")
        return self.test_examples

    def get_dep_type_list(self, feature_data, feature_type='dep'):
        feature2count = defaultdict(int)
        for sent in feature_data:
            for item in sent:
                pos = item[feature_type]
                if self.direct:
                    # direct
                    feature_in = pos + "_in"
                    feature_out = pos + "_out"
                    feature2count[feature_in] += 1
                    feature2count[feature_out] += 1
                else:
                    # undirect
                    feature2count[pos] += 1
        feature2id = {"none": 0, "self_loop": 1}
        for key in feature2count:
            feature2id[key] = len(feature2id)
        dep_type_list = feature2id.keys()
        logging.info(dep_type_list)
        return feature2id

    def get_knowledge_feature(self, data_dir, feature_type='dep', flag="train"):
        sfp = StanfordFeatureProcessor(data_dir)
        feature_data = sfp.read_features(flag=flag)
        feature_data = filter_useful_feature(feature_data, feature_type=feature_type, direct=self.direct)
        feature2id = self.get_dep_type_list(feature_data, feature_type)
        for dep,id in feature2id.items():
            if dep not in self.feature2id:
                self.feature2id[dep] = len(self.feature2id)
        return feature_data

    def get_feature2id_dict(self):
        return self.feature2id

    def get_labels(self):
        return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS', 'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU', "[CLS]", "[SEP]"]

    def _create_examples(self, features, set_type):
        examples = []
        class_count = np.zeros(3)
        for i, feature in enumerate(features):
            guid = "%s-%s" % (set_type, i)
            text_a = [x['word'] for x in feature]
            label = [x['tag'] for x in feature]
            dep = [x['dep'] for x in feature]
            adj = [x['adj'] for x in feature]
            dep_text = [x['dep_text'] for x in feature]
            label = ot2bieos_ts(label)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label, dep=dep, adj=adj, dep_text=dep_text))
            gold_ts = tag2ts(ts_tag_sequence=label)
            for (b, e, s) in gold_ts:
                if s == 'POS':
                    class_count[0] += 1
                if s == 'NEG':
                    class_count[1] += 1
                if s == 'NEU':
                    class_count[2] += 1
            assert len(text_a) == len(label)
            assert len(text_a) == len(dep)
            assert len(text_a) == len(adj)
            assert len(text_a) == len(dep_text)
            if i < 4:
                logging.info("text: {}".format(",".join(text_a)))
                logging.info("label: {}".format(",".join(label)))
                logging.info("dep: {}".format(",".join(dep)))
                logging.info("adj: {}".format(";".join([','.join([str(_) for _ in x]) for x in adj])))
                logging.info("dep_text: {}".format(";".join([','.join(x) for x in dep_text])))
        print("%s class count: %s" % (set_type, class_count))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, feature2id):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    b_use_valid_filter = False
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
                    b_use_valid_filter = True
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length


        adj_matrix = [[0] * max_seq_length for _ in range(max_seq_length)]
        for i, adj in enumerate(example.adj):
            for j,dep in enumerate(adj):
                adj_matrix[i+1][j+1] = dep
        for i in range(len(adj_matrix)):
            adj_matrix[i][i] = 1
        dep_matrix = [[0] * max_seq_length for _ in range(max_seq_length)]
        for i, dep_text in enumerate(example.dep_text):
            for j, dep in enumerate(dep_text):
                dep_matrix[i + 1][j + 1] = feature2id.get(dep,0)

        if ex_index < 2:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label: %s (id = %s)" % (",".join([str(x) for x in example.label]), ",".join([str(x) for x in label_ids])))
            logging.info("valid: %s" % " ".join([str(x) for x in valid]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask,
                          b_use_valid_filter=b_use_valid_filter,
                          adj_matrix=adj_matrix,
                          dep_matrix=dep_matrix))
    return features



def load_examples(args, tokenizer, processor, label_list, mode):
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
        knowledge_feature2id = processor.get_feature2id_dict()
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
        knowledge_feature2id = processor.get_feature2id_dict()
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
        knowledge_feature2id = processor.get_feature2id_dict()

    features = convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer, knowledge_feature2id)
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(examples))
    logging.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_b_use_valid_filter = torch.tensor([f.b_use_valid_filter for f in features], dtype=torch.bool)
    all_adj_matrix = torch.tensor([f.adj_matrix for f in features], dtype=torch.long)
    all_dep_matrix = torch.tensor([f.dep_matrix for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_label_mask,
                         all_b_use_valid_filter, all_adj_matrix, all_dep_matrix)

def save_zen_model(save_zen_model_path, model, args):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_zen_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_zen_model_path, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    with open(output_config_file, "w", encoding='utf-8') as writer:
        writer.write(model_to_save.config.to_json_string())
    output_args_file = os.path.join(save_zen_model_path, 'training_args.bin')
    torch.save(args, output_args_file)


