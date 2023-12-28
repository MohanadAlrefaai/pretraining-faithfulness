import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version


from data import DataCollatorForSeq2SeqWithMultipleReferences
from BSF_Trainer import BSFTrainer
from trainer import CustomTrainer

import traceback


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.21.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

import string
import random


import pandas as pd
import spacy
import random    
import nltk

def process_ds(dataset, output_dir, num_proc = 1, ranges = range(100), batch_size = 1000):
    
    def max_length(dataset, num_proc, ranges, batch_size):
        x = []
        def process(examples):
            articles = []
            for e in examples["article"]:
                articles.append(
                    " ".join(e.split()[0:1024])
                )

            examples["article"] = articles
        
            return examples
        
        return dataset.select(ranges).map(
            process,
            batched=True,
            num_proc=num_proc,
            batch_size=batch_size
        )
    
    def gpu_spacy_process(dataset, ranges = range(100), batch_size = 1000):

        from thinc.api import set_gpu_allocator, require_gpu

        import spacy
        def process(examples):

            require_gpu(0)
            # Use the GPU, with memory allocations directed via PyTorch.
            # This prevents out-of-memory errors that would otherwise occur from competing
            # memory pools.
            #set_gpu_allocator("pytorch")
            #require_gpu(1 if random.random() < 0.3 else 0)
            try:
                spacy_pipeline = spacy.load('en_core_web_sm', enable=["ner"])
            except OSError:
                logging.warning("Downloading language model for the spaCy model.")
                from spacy.cli import download
                download('en_core_web_sm')
                spacy_pipeline = spacy.load('en_core_web_sm')

            texts = examples["article"]
            list_entities = []

            spacy_docs = spacy_pipeline.pipe(texts)
            for doc in spacy_docs:
                list_entities.append([a.text for a in doc.ents])

            examples["ner"] = list_entities
            return examples

        return dataset.select(ranges).map(
            process,
            batched=True,
            num_proc=2,
            batch_size=batch_size
        )
    
    def pre_process_dataset(dataset, num_proc = 1, ranges = range(100), batch_size = 1000):
        import string
        import random


        import pandas as pd
        import random    
        import nltk
        import math

        import evaluate
        import numpy as np

        from datasets import load_dataset, load_metric
        data_args = {
            "ner_mlm": True,
            "ner_sgs_mlm": False,
            "ner_mlm_prob": 0.35
        }

        NER_MASK = "<mask1>"
        NER_TOKEN_MASK = "<mask2>"
        MLM_CONNECTOR = "<conn1>"
        MLM_SGS_CONNECTOR = "<conn2>"

        text_column = "article"
        summary_column = "abstract"

        #bertscore = load_metric('bertscore')

        rouge = evaluate.load("rouge")


        def get_other_text(sentence, text):
            res = text.replace(sentence, "", 1)
            if (len(res) == 0 and len(res.split()) == 0):
                res = text
            return res

        def extract_entties_with_spacy(source, spacy_pipeline):

            list_entities = []

            spacy_doc = spacy_pipeline(source)#, n_process=-1)
            list_entities = [a.text for a in spacy_doc.ents]

            return list_entities


        def mask_sentence(sentence, entities, prob_ner, prob_token):

            tokens_sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            processed = sentence
            tokens_sentnece_list = tokens_sentence.split()

            tokens_sentnece_list_masking = np.zeros(len(tokens_sentnece_list))

            for entity in entities:
                if entity in sentence:
                    tokens_sentence = tokens_sentence.replace(entity, "")
                    if random.random() <= prob_ner:
                        entity_tokens = entity.split()
                        entity_length = len(entity_tokens)
                        for tokens_index in range(len(tokens_sentnece_list)):
                            tokens_seq = " ".join(tokens_sentnece_list[tokens_index:tokens_index + entity_length])
                            entity_seq = entity
                            if tokens_seq == entity_seq:
                                tokens_sentnece_list_masking[tokens_index:tokens_index + entity_length] = 2

            for token_index in range(len(tokens_sentnece_list)):

                token = tokens_sentnece_list[token_index]

                if tokens_sentnece_list_masking[token_index] == 2:
                    # NER MASK
                    continue

                if random.random() <= prob_token:
                    tokens_sentnece_list_masking[token_index] = 1
                else :
                    tokens_sentnece_list_masking[token_index] = 0

            result = []

            for i in range(len(tokens_sentnece_list_masking)):
                if tokens_sentnece_list_masking[i] == 2:
                    # if prev is mask2 then do not repeat
                    if i > 0 and tokens_sentnece_list_masking[i - 1] == 2:
                        continue

                    result.append(NER_MASK)

                elif tokens_sentnece_list_masking[i] == 1:
                    # if prev is mask1 then do not repeat
                    if i > 0 and tokens_sentnece_list_masking[i - 1] == 1:
                        result.append(NER_TOKEN_MASK)

                else:
                    result.append(tokens_sentnece_list[i])

            processed = " ".join(result)

            label = " ".join(np.where(tokens_sentnece_list_masking, tokens_sentnece_list, "<unmasked>"))

            return [processed, label, tokens_sentnece_list_masking]

        def sentence_scorer(sentence, text, entities, rouge_score):

            # count entities
            entites_count = 0
            for entity in entities:
                if entity in sentence:
                    entites_count += 1
            entites_score = (entites_count * 1.0) / 4

            # bertscore
            #bertscore_results = bertscore.compute(predictions=[sentence], references=[text], lang='en')

    #        bertscore_value = np.average(bertscore_results["f1"])

            return entites_score + rouge_score# + bertscore_value


        def to_sentences(text):
            sentences = nltk.sent_tokenize(text)
            texts = [get_other_text(sent, text) for sent in sentences]

            return (texts, sentences)

        def custom_train_preprocess(examples):


            list_entities = []


            all_texts = []
            all_sentences = []
            groups = []
            for i in range(len(examples[text_column])):

                preprocessed_exp = examples[text_column][i]
                
                
                texts, sentences = to_sentences(preprocessed_exp)
                start_index = len(all_texts)
                
                all_texts.extend(texts)
                all_sentences.extend(sentences)
                end_index = len(all_texts)
                groups.append((start_index, end_index, sentences))

            # compute rouge score for the whole batch

            result_rouge = rouge.compute(predictions=all_sentences, references=all_texts, rouge_types=["rouge1"], use_aggregator=False)
            rouge_scores = [score for score in result_rouge.get("rouge1")]


            documents = []
            summaries = []
            mlm_labels = []
            masked_docuemnts = []
            for i in range(len(examples[text_column])):
                
                document = ""
                summary = ""
                masked_document = ""
                mlm_label = ""
                
                preprocessed_exp = examples[text_column][i]
                
                if len(preprocessed_exp) == 0:
                    mlm_labels.append("")
                    summaries.append(examples[summary_column][i])
                    documents.append(preprocessed_exp)
                    masked_docuemnts.append("")
                    continue

                list_entities = examples["ner"][i]


                try:
                    
                    sentences = groups[i][2]
                    this_rouge_scores = rouge_scores[groups[i][0]:groups[i][1]]
                except:
                    print(i)
                    print(groups[i])
                    raise("error")

                examples_data = {
                    "sentence": sentences,
               #     "rouge1": rouge_scores
                    "rouge1": this_rouge_scores
                }

                df = pd.DataFrame(examples_data)
                df["score"] = df.apply(lambda x: sentence_scorer(x["sentence"], preprocessed_exp, list_entities, x["rouge1"]), axis=1)
                df.sort_values("score", ascending=False, inplace=True)
                df["normalized_score"] = df["score"].map(lambda x: x / (df["score"].max() + 0.001))
                masked_labels = df.apply(
                    lambda x: 
                        mask_sentence(x["sentence"], list_entities, data_args["ner_mlm_prob"], x["normalized_score"] * (0.1 + data_args["ner_mlm_prob"]))
                              , 
                    axis=1)
                
                try:
#                    print(masked_labels)
                    df["masked_labels"] = masked_labels
                except:
                    print(i)
                    print("preprocessed", preprocessed_exp, "end")
                    raise 'error'
                    
                df["masked"] = df["masked_labels"].map(lambda x: x[0] )
                df["labels"] = df["masked_labels"].map(lambda x: x[1])


                if data_args["ner_mlm"]:
                    # mlm for faithfull
                    num_sentences = 0
                    check_words = ""
                    for index in df.index:
                        check_words = " ".join([check_words, df.loc[index]["sentence"]])
                        num_sentences += 1
                        if (len(check_words.split()) > 512):
                            break

                    selectedDF = df.head(num_sentences).sort_index()
                    document =  " ".join(selectedDF["sentence"])
                    summary =  examples[summary_column][i]
                    masked_document =  " ".join(selectedDF["masked"])
                    mlm_label =  " ".join(selectedDF["labels"])
                   
                if data_args["ner_mlm"]:
                    masked_document = " ".join([MLM_CONNECTOR, masked_document])
                
                mlm_labels.append(mlm_label)
                masked_docuemnts.append(masked_document)
                summaries.append(summary)
                documents.append(document)

            new_examples = {}
            new_examples[text_column] = documents
            new_examples[summary_column] = summaries
            new_examples["mlm_label"] = mlm_labels
            new_examples["masked_document"] = masked_docuemnts
            return new_examples

        return dataset.select(ranges).map(
            custom_train_preprocess,
            batched=True,
            num_proc=num_proc,
            batch_size=batch_size
        )
    
    maxed_length_dataset = max_length(dataset, num_proc, ranges, batch_size)
    
    ner_dataset = gpu_spacy_process(maxed_length_dataset, range(len(maxed_length_dataset)), batch_size)
    
    processed_dataset = pre_process_dataset(ner_dataset, num_proc, range(len(ner_dataset)), batch_size)
    
    processed_dataset.save_to_disk(output_dir)   
    
    return processed_dataset


dataset = load_dataset(
    "ccdv/arxiv-summarization",
)
output_dir = r"G:\.cache\huggingface\datasets\mlm\arxiv_1024"
process_ds(
    dataset["train"], 
    output_dir, 
    10, 
    range(len(dataset["train"])), 
    1000
)