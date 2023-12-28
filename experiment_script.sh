# Preprocess
python src/preprocess.py
# Custom MLM Train
# Arxiv
python src/run_mlm_custom.py --overwrite_output_dir --text_column masked_document --summary_column mlm_label --num_train_epochs 1 --do_eval --dataset_name arxiv-1024 --ner_mlm True --preprocessing_num_workers 8  --per_device_eval_batch_size 1 --per_device_train_batch_size 1 --fp16 --model_name_or_path facebook/bart-large  --do_train  --max_source_length 1024 --max_target_length 1024 --save_strategy steps --save_steps 10000  --evaluation_strategy epoch --logging_strategy steps --logging_steps 500  --output_dir out_arxiv_mlm_train_1024 --remove_unused_columns true --warmup_steps 5000 --learning_rate 2e-5 --gradient_accumulation_step 1 --label_smoothing 0.1
# XSUM
python src/run_mlm_custom.py  --num_train_epochs 1 --do_eval --dataset_name xsum --ner_mlm True --preprocessing_num_workers 8  --per_device_eval_batch_size 8 --per_device_train_batch_size 8 --fp16 --model_name_or_path facebook/bart-large-xsum  --do_train  --max_source_length 512 --max_target_length 512 --save_strategy steps --save_steps 1500  --evaluation_strategy steps --eval_steps 1500 --logging_strategy steps --logging_steps 100  --output_dir out_xsum_xsum_mlm_train_512 --remove_unused_columns true --warmup_steps 1000 --learning_rate 2e-5 --gradient_accumulation_step 1

# Finetuning
python src/run_summarization_custom.py  --num_train_epochs 1 --do_train --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --include_inputs_for_metrics --fp16 --dataset_name ccdv/arxiv-summarization --preprocessing_num_workers 8   --model_name_or_path out_xsum_xsum_mlm_train_512/checkpoint-34500  --do_eval  --max_source_length 1024 --max_target_length 128 --save_strategy steps --save_steps 10000  --output_dir out_arxiv_xsun_pretrain_train_1024_128_b8_beams6 --num_beams 6 --remove_unused_columns true --predict_with_generate --learning_rate 2e-5 --gradient_accumulation_step 1 --warmup_steps 5000 --label_smoothing 0.1

# Pred
python src/run_summarization_custom.py  --do_predict --per_device_eval_batch_size 8 --fp16 --dataset_name ccdv/arxiv-summarization  --preprocessing_num_workers 8   --model_name_or_path out_arxiv_xsun_pretrain_train_512_128_b8_beams1  --max_source_length 512 --max_target_length 128 --save_strategy steps  --output_dir predict_out_arxiv_xsum_pretrain_train_512_128_8b --num_beams 6 --remove_unused_columns true --predict_with_generate --include_inputs_for_metrics
# QE
python src/run_qe.py --exp predict_out_arxiv_xsum_pretrain_train_512_128_8b
