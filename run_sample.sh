#train
python dgsa_main.py --do_train --do_eval --gcn_layer_number 1 --data_dir data/sample_data --task_name sample_data --init_checkpoint ./bert-large-uncased/ --vocab_file ./bert-large-uncased/vocab.txt --max_seq_length=128 --do_lower_case --train_batch_size=32 --eval_batch_size=16 --num_train_epochs 30 --learning_rate 3e-5 --warmup_proportion 0.06  --output_dir results/ --fp16 --seed 40

#test
python dgsa_main.py --do_test --eval_model ./results/DGSA.sample_data.BERT.L --data_dir ./data/sample_data --gcn_layer_number 1  --do_lower_case
