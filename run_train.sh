#!/bin/bash


#LAPTOP
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/LAPTOP --task_name LAPTOP --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#REST
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/REST --task_name REST --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER1
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER1 --task_name TWITTER1 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER2
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER2 --task_name TWITTER2 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER3
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER3 --task_name TWITTER3 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER4
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER4 --task_name TWITTER4 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER5
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER5 --task_name TWITTER5 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER6
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER6 --task_name TWITTER6 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER7
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER7 --task_name TWITTER7 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER8
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER8 --task_name TWITTER8 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER9
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER9 --task_name TWITTER9 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#TWITTER10
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/TWITTER10 --task_name TWITTER10 --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40
