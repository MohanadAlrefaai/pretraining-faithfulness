import pandas as pd
import os 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--exp", dest="exp_dir",)

parser.add_argument("--gpu_id", dest="gpu_id", type=int, default=-1,)

args = parser.parse_args()

if (args.gpu_id != -1):
    import torch
    torch.cuda.set_device(args.gpu_id)

dir_path = os.path.dirname(os.path.realpath(__file__))
exp_dir  = args.exp_dir
# exp_dir  = "out_cnn_full_pred"

dir_name = os.path.join(dir_path, "../", exp_dir)

file_name = os.path.join(dir_name, "pairs_data.json")
pred_data = pd.read_json(file_name)

# Metrics
from FEQA.feqa_score import feqa_score
#metric = load_metric("rouge")
#bertscore = load_metric('bertscore')

print("computing FEQA")
feqa_results = feqa_score(pred_data['inputs'], pred_data['preds'])

result = { "feqa_value": 0 }

result["feqa_value"] = feqa_results

import json
with open(os.path.join(exp_dir, "feqa_result.json"), 'w') as fp:
    json.dump(result, fp)