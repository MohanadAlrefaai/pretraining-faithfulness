import pandas as pd
import os 
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument("-e", "--exp", dest="exp_dir",
                    help="write report to FILE", metavar="FILE")

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
from QuestEval.questeval_metric import QuestEval
#from FEQA.feqa_score import feqa_score
#metric = load_metric("rouge")
#bertscore = load_metric('bertscore')
current_dir = os.path.dirname(os.path.realpath(__file__))
qe_logdir = os.path.join(current_dir, "..", exp_dir, "logs")
print("qe_logdir", qe_logdir)
if not os.path.exists(qe_logdir):
    os.makedirs(qe_logdir)
questeval = QuestEval(log_dir=qe_logdir)
print("computing QuestEval")
questeval_results = questeval.corpus_questeval(pred_data['inputs'], pred_data['preds'])

result = { "qe_value": 0 }

result["qe_value"] = questeval_results['corpus_score']
print("result", result)
import json
with open(os.path.join(dir_name, "qe_result.json"), 'w') as fp:
    json.dump(result, fp)