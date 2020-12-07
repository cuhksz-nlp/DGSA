from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import datetime
import time
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from BERT import WEIGHTS_NAME
from BERT import VOCAB_NAME
from BERT import BertTokenizer
from BERT import BertAdam
from BERT import LinearWarmUpScheduler
from dgsa_model import DGSA
from data_utils import E2EASAOTProcessor, convert_examples_to_features, save_zen_model, load_examples
from dgsa_eval import evaluate_ote
from apex import amp
from BERT import is_main_process

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main_train(args, model, tokenizer, processor, label_list, device, n_gpu):
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        print("using fp16")
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)

        if args.loss_scale == 0:

            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale=args.loss_scale)
        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                          total_steps=num_train_optimization_steps)
    else:
        print("using fp32")
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    average_loss = 0

    print("data prep")
    train_data = load_examples(args, tokenizer, processor, label_list, "train")

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    nb_tr_examples = 0
    for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):
        if args.max_steps > 0 and global_step > args.max_steps:
            break
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if args.max_steps > 0 and global_step > args.max_steps:
                break
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask, b_use_valid_filter, \
            adj_matrix, dep_matrix = batch

            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids,
                         valid_ids=valid_ids, adjacency_matrix=adj_matrix)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            average_loss += loss
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if args.fp16:
                    # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                    scheduler.step()

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                logging.info("Global Steps:{} Final Loss = {}".format(global_step, average_loss))
                average_loss = 0

        if args.local_rank == -1 or torch.distributed.get_rank() == 0 or args.world_size <= 1:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "epoch-{}".format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_zen_model(output_dir, model, args)

    loss = tr_loss / nb_tr_steps if args.do_train else None
    return loss, global_step

def evaluate(args, model, tokenizer, processor, label_list, device, mode="test"):
    num_labels = len(label_list) + 1
    eval_data = load_examples(args, tokenizer, processor, label_list, mode)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = 'unknown'
    nb_tr_examples, nb_tr_steps = 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask, b_use_valid_filter,\
            adj_matrix, dep_matrix = batch

        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         valid_ids=valid_ids, adjacency_matrix=adj_matrix)

        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])

    logger.info("nb_tr_examples: {}, nb_tr_steps: {}".format(nb_tr_examples, nb_tr_steps))

    result = evaluate_ote(y_true, y_pred)
    logging.info(result)

    return {
        "precision": result[0],
        "recall": result[1],
        "f1": result[2]
    }


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default="E2EASA",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="results",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The checkpoint file from pretraining")
    parser.add_argument("--eval_model",
                        default=None,
                        type=str,
                        help="The model for evaluation")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_dev",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_all_cuda",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--google_pretrained",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size",
                        type=int,
                        default=0)
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--gcn_layer_number', type=int, default=3)

    args = parser.parse_args()

    return args

def train(args):
    if 'SLURM_NTASKS' in os.environ:
        args.world_size = int(os.environ['SLURM_NTASKS'])
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
    if 'SLURM_LOCALID' in os.environ:
        args.local_rank = int(os.environ['SLURM_LOCALID'])

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                             world_size=args.world_size)
    args.device = device
    args.n_gpu = n_gpu
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        str_args = "bts_{}_lr_{}_warmup_{}_seed_{}_gcn_layer_number_{}".format(
            args.train_batch_size,
            args.learning_rate,
            args.warmup_proportion,
            args.seed,
            args.gcn_layer_number
        )
        args.output_dir = os.path.join(args.output_dir, 'result-{}-{}-{}'.format(args.task_name, str_args, now_time))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    processor = E2EASAOTProcessor()
    type_num = processor.get_type_num()
    label_list = processor.get_labels()
    num_labels = processor.get_label_num()

    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512)
    print("USED CHECKPOINT from", args.init_checkpoint)
    model = DGSA.from_pretrained(args.init_checkpoint, num_labels=num_labels, type_num=type_num,
                                        gcn_layer_number=args.gcn_layer_number)
    model.to(device)

    results = {"init_checkpoint": args.init_checkpoint, "lr": args.learning_rate, "warmup": args.warmup_proportion,
               "train_batch_size": args.train_batch_size * args.gradient_accumulation_steps * args.world_size,
               "fp16": args.fp16, "gcn_layer_number":args.gcn_layer_number}
    if args.do_train:
        results["train_start_runtime"] = time.time()
        loss, global_step = main_train(args, model, tokenizer, processor, label_list, device, n_gpu)
        results["train_runtime"] = time.time() - results["train_start_runtime"]
        results["global_step"] = global_step
        results["loss"] = loss
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        results["eval_start_runtime"] = time.time()
        if args.eval_all_cuda:
            args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        results["best_checkpoint"] = 0
        results["best_f1_score"] = 0
        results["best_checkpoint_path"] = ""
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = DGSA.from_pretrained(checkpoint, num_labels=num_labels, type_num=type_num,
                                                gcn_layer_number=args.gcn_layer_number)
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            if args.eval_dev:
                result = evaluate(args, model, tokenizer, processor, label_list, device, mode="dev")
            else:
                result = evaluate(args, model, tokenizer, processor, label_list, device, mode="test")
            if result["f1"] > results["best_f1_score"]:
                results["best_f1_score"] = result["f1"]
                results["best_checkpoint"] = global_step
                results["best_checkpoint_path"] = checkpoint
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        if args.eval_dev:
            model = DGSA.from_pretrained(results["best_checkpoint_path"], num_labels=num_labels,
                                                type_num=type_num, gcn_layer_number=args.gcn_layer_number)
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            result = evaluate(args, model, tokenizer, processor, label_list, device, mode="test")
            result = {"test_{}".format(k): v for k, v in result.items()}
            results.update(result)
        results["eval_runtime"] = time.time() - results["eval_start_runtime"]
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write(json.dumps(results, ensure_ascii=False))
        for key in sorted(results.keys()):
            logger.info("{} = {}\n".format(key, str(results[key])))

def test(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    processor = E2EASAOTProcessor()
    type_num = processor.get_type_num()
    label_list = processor.get_labels()
    num_labels = processor.get_label_num()

    tokenizer = BertTokenizer(os.path.join(args.eval_model, VOCAB_NAME), do_lower_case=args.do_lower_case, max_len=512)
    logger.info("USED CHECKPOINT from {}".format(args.eval_model))
    model = DGSA.from_pretrained(args.eval_model, num_labels=num_labels, type_num=type_num,
                                 gcn_layer_number=args.gcn_layer_number)
    model.to(args.device)
    result = evaluate(args, model, tokenizer, processor, label_list, args.device, mode="test")
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        writer.write(json.dumps(result, ensure_ascii=False))
    for key in sorted(result.keys()):
        logger.info("{} = {}\n".format(key, str(result[key])))

def main():
    args = get_args()

    if args.do_train:
        train(args)
    elif args.do_eval:
        test(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval` must be True.')

if __name__ == "__main__":
    main()
