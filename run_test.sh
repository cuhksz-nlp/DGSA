#!/bin/bash


#LAPTOP
python dgsa_main.py --do_eval --eval_model ./release/DGSA.LAPTOP.BERT.L --data_dir ./data/LAPTOP --gcn_layer_number 1  --do_lower_case

#REST
python dgsa_main.py --do_eval --eval_model ./release/DGSA.REST.BERT.L --data_dir ./data/REST --gcn_layer_number 1  --do_lower_case

#TWITTER1
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER1.BERT.L --data_dir ./data/TWITTER1 --gcn_layer_number 1  --do_lower_case

#TWITTER2
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER2.BERT.L --data_dir ./data/TWITTER2 --gcn_layer_number 1  --do_lower_case

#TWITTER3
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER3.BERT.L --data_dir ./data/TWITTER3 --gcn_layer_number 1  --do_lower_case

#TWITTER4
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER4.BERT.L --data_dir ./data/TWITTER4 --gcn_layer_number 1  --do_lower_case

#TWITTER5
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER5.BERT.L --data_dir ./data/TWITTER5 --gcn_layer_number 1  --do_lower_case

#TWITTER6
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER6.BERT.L --data_dir ./data/TWITTER6 --gcn_layer_number 1  --do_lower_case

#TWITTER7
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER7.BERT.L --data_dir ./data/TWITTER7 --gcn_layer_number 1  --do_lower_case

#TWITTER8
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER8.BERT.L --data_dir ./data/TWITTER8 --gcn_layer_number 1  --do_lower_case

#TWITTER9
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER9.BERT.L --data_dir ./data/TWITTER9 --gcn_layer_number 1  --do_lower_case

#TWITTER10
python dgsa_main.py --do_eval --eval_model ./release/DGSA.TWITTER10.BERT.L --data_dir ./data/TWITTER10 --gcn_layer_number 1  --do_lower_case