import torch
from torch import nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
import numpy as np
from datasets import load_dataset, load_metric
from torch.nn import Softmax

class BSFTrainer(Seq2SeqTrainer):
    """
    Custom trainer for multiple Cross Entropy Loss. Adapted from original huggingface trainer code.
    """
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        alpha = 1.0,
        bertscore=None
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.alpha = alpha
        #metric = load_metric("rouge")
        bertscore = load_metric('bertscore')
        self.bertscore = bertscore

    def compute_loss(self, model, inputs, return_outputs=False):
        #preds, labels, inputs = eval_preds
        

        

        #preds = model.generate(**inputs, num_beams=self.num_beams)
        original_loss, output = super().compute_loss(model, inputs, return_outputs=True)
        preds = torch.argmax(F.log_softmax(output.get("logits"), dim=-1), dim= -1)
        if isinstance(preds, tuple):
            preds = preds[0]

        #print("preds", preds)
        preds = torch.where(preds >= 0, preds, self.tokenizer.pad_token_id)
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        inputs_ids = torch.where(inputs["input_ids"] >= 0, inputs["input_ids"], self.tokenizer.pad_token_id)
        
        decoded_inputs = self.tokenizer.batch_decode(inputs_ids, skip_special_tokens=True)

        bertscore_fact_results = self.bertscore.compute(predictions=decoded_preds, references=decoded_inputs, lang='en')
        #bsfact_loss = torch.mean(torch.square(torch.div(2, torch.FloatTensor(bertscore_fact_results["precision"]) + 1)))
        bsfact_loss = -1 * torch.mean(torch.FloatTensor(bertscore_fact_results["precision"]))
        return original_loss*0.0 + 1*bsfact_loss
        return original_loss*0.4 + 0.6*bsfact_loss
        return 
        # first get encoder outputs to save computation
        encoder_outputs = model.get_encoder()(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"]
        )
        # encoder_outputs = model(
        #     input_ids = inputs["input_ids"],
        #     attention_mask = inputs["attention_mask"]
        # )
        
        inputs["encoder_outputs"] = encoder_outputs.get('logits')

        # Cross Entropy Loss

        # original XE
        orig_loss = super().compute_loss(model, inputs, return_outputs)
        loss = orig_loss
        
        # additional labels
        if additional_labels is not None:
            # compute loss for labels and additional_labels separaetly

            inputs["decoder_input_ids"] = additional_decoder_input_ids
            inputs["labels"] = additional_labels
            additional_loss = super().compute_loss(model, inputs, return_outputs)
            
            loss += self.alpha * additional_loss
        
        return loss