"""
Fine-tuning script for summarization adapted from Huggingface (https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py).
"""

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


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    alpha: float = field(default=1.0)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    reference_file: Optional[str] = field(
        default=None
    )
    additional_reference_file: Optional[str] = field(
        default=None
    )
    
    gpu_id: Optional[int] = field(
        default=-1
    )

    ner_mlm: Optional[bool] = field(
        default=True
    )

    ner_mlm_prob: Optional[float] = field(
        default=0.6
    )

    ner_sgs_mlm: Optional[bool] = field(
        default=False
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser(Seq2SeqTrainingArguments)
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    from datasets import load_from_disk

    wiki_datasets = load_from_disk("G:\.cache\huggingface\datasets\wiki_100_1000")

    # sample = wiki_datasets["train"].select(range(10)).map(
    #     custom_train_preprocess,
    #     batched=True,
    #     num_proc=2,#data_args.preprocessing_num_workers,
    # )

   # if data_args.max_train_samples is not None:
   #     max_train_samples = min(len(train_dataset), data_args.max_train_samples)
   #     train_dataset = train_dataset.select(range(max_train_samples))
    # with training_args.main_process_first(desc="train dataset map pre-processing"):
        
    #from transformers.training_args import  main_process_first

    import pandas as pd   
    import nltk
    import numpy as np
    import string
    import random

    data_args = {
        "ner_mlm": False,
        "ner_sgs_mlm": True,
        "ner_mlm_prob": 0.6
    }

    NER_MASK = "<mask1>"
    NER_TOKEN_MASK = "<mask2>"
    MLM_CONNECTOR = "<conn1>"
    MLM_SGS_CONNECTOR = "<conn2>"

    
    text_column = "text"
    summary_column = "summary"

    from datasets import load_dataset, load_metric 

    rouge = load_metric("rouge")

    import spacy
    import math

    try:
        spacy_pipeline = spacy.load('en_core_web_sm')
    except OSError:
        logging.warning("Downloading language model for the spaCy model.")
        from spacy.cli import download
        download('en_core_web_sm')
        spacy_pipeline = spacy.load('en_core_web_sm')

    def custom_train_preprocess(examples, args):
         

        bertscore = load_metric('bertscore')

        def extract_entties_with_spacy(source, ):


            list_entities = []

            spacy_doc = spacy_pipeline(source)
            list_entities = [a.text for a in spacy_doc.ents]
            
            return list_entities

        def mask_sentence(sentence, entities, prob_ner, prob_token):



            tokens_sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            processed = sentence
            
            
            for entity in entities:
                if entity in sentence:
                    tokens_sentence = tokens_sentence.replace(entity, "")
                    if random.random() <= prob_ner:
                        processed = sentence.replace(entity, NER_MASK)
                        
            for token in tokens_sentence.split():
                if token == NER_MASK:
                    continue
                if random.random() <= prob_token:
                    processed = processed.replace(" " + token, " " + NER_TOKEN_MASK, 1)
                    processed = processed.replace(" " + token + " ", " " + NER_TOKEN_MASK + " ", 1)
                    processed = processed.replace(token + " ", NER_TOKEN_MASK + " ", 1)

            for i in range(0, len(tokens_sentence.split())):
                multiple_masks = NER_TOKEN_MASK
                for j in range (0, i):
                    multiple_masks += " " + NER_TOKEN_MASK
                processed = processed.replace(multiple_masks, NER_TOKEN_MASK)
            
            return processed
        
        def sentence_scorer(sentence, text, entities, rouge, bertscore):

            text_else = text.replace(sentence, "", 1)
            if len(text_else) == 0:
                text_else = text
            
            # count entities
            entites_count = 0
            for entity in entities:
                if entity in sentence:
                    entites_count += 1
            entites_score = (entites_count * 1.0) / 4

            # rouge score
            #result_rouge = rouge.compute(predictions=[sentence], references=[text_else], use_stemmer=True)
            #rouge_score = result_rouge.get("rouge1").mid.fmeasure
            rouge_score = 0.5

            # bertscore
            #bertscore_results = bertscore.compute(predictions=[sentence], references=[text_else], lang='en')
            
            #bertscore_value = bertscore
            # bertscore_value = np.average(bertscore_results["f1"])
            bertscore_value = 0.5
            
            return entites_score + rouge_score + bertscore_value

        if (not data_args["ner_mlm"]) and (not data_args["ner_sgs_mlm"]):
            return examples

        list_entities = []
        
        documents = []
        summaries = []
        for i in range(len(examples[text_column])):

            preprocessed_exp = examples[text_column][i]

            list_entities = extract_entties_with_spacy(preprocessed_exp)
            
            sentences = nltk.sent_tokenize(preprocessed_exp)
            def get_other_text(sentence):
                res = preprocessed_exp.replace(sentence, "", 1)
                if (len(res) == 0 and len(res.split()) == 0):
                    res = preprocessed_exp
                return res

            other_texts = [get_other_text(sent) for sent in sentences]
            # bert_scores = bertscore.compute(predictions=sentences, references=other_texts, lang='en')

            examples_data = {
                "sentence": sentences,
                # "bert_score": bert_scores["f1"]
            }
            #bertscore = {}
            
            df = pd.DataFrame(examples_data)
            df["score"] = df.apply(lambda x: sentence_scorer(x["sentence"], preprocessed_exp, list_entities, rouge, 0.5), axis=1)
            df.sort_values("score", ascending=False, inplace=True)
            df["normalized_score"] = df["score"].map(lambda x: x / df["score"].max())
            df["masked"] = df.apply(lambda x: mask_sentence(x["sentence"], list_entities, data_args["ner_mlm_prob"], x["normalized_score"] * data_args["ner_mlm_prob"]), axis=1)
            df["num_masked1"] = df["masked"].map(lambda x: x.count("<mask1>"))
            df["num_masked2"] = df["masked"].map(lambda x: x.count("<mask2>"))
            df["length"] = df["sentence"].map(lambda x: len(x.split()))
            df["length_masked"] = df["masked"].map(lambda x: len(x.split()))
            df["masked_prob"] = df.apply(lambda x: ((x["num_masked2"] + x["num_masked1"]) + (x["length"] - x["length_masked"])) / x["length"], axis=1)
            
            if data_args["ner_mlm"]:
                # mlm for faithfull
                summary = " ".join(df["sentence"].sort_index( ))
            elif data_args["ner_sgs_mlm"]:
                # mlm-sgs for faithfull
                summary = " ".join(df.head(math.ceil(df.shape[0] / 3.0)).sort_index( )["sentence"])
                
            preprocessed_exp = " ".join(df["masked"].sort_index())
            if data_args["ner_mlm"]:
                preprocessed_exp = " ".join([MLM_CONNECTOR, preprocessed_exp])
            elif  data_args["ner_sgs_mlm"]:
                preprocessed_exp = " ".join([MLM_CONNECTOR, MLM_SGS_CONNECTOR, preprocessed_exp])
            
                    
        # print(df.sort_index())
        # print(df.describe())
                
            summaries.append(summary)
            documents.append(preprocessed_exp)
            
        new_examples = {}
        new_examples[text_column] = documents
        new_examples[summary_column] = summaries
        return new_examples

    # with training_args[0].main_process_first(desc="train dataset map pre-processing"):
    sample = wiki_datasets["train"].select(range(1000)).map(
        lambda x: custom_train_preprocess(x, {}),
        batched=True,
        num_proc=4,#data_args.preprocessing_num_workers,
        desc="Running custom preprocesssing on train dataset",
    )

    print(sample)


if __name__ == "__main__":
    main()