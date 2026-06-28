import warnings
import os
import torch
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import data_loader, SquadEvaluator, TASTEset
import pandas as pd
import re
from aligner_pt import AlignerLoss, BertAligner
import argparse
import json

def main(args):
    data_test = TASTEset.from_disk(args.data)
    data_langs = re.search(r'([a-z]{2}-[a-z]{2})(-[a-z]{2})?', args.data).group(0)
    batch_size = 128
    dataset_test = data_loader(data_test, 
                        batch_size,
                        # n_rows=100
                        )
    print("langs:", data_langs)
    model_list = []
    print('model_dir:', args.model_dir)
    for model_path, dirs, files in os.walk(args.model_dir):
        # print(model_path, files)
        if files:
            model_langs = re.search(r'_([a-z]{2}-[a-z]{2}(-[a-z]{2})?)_', model_path).group(1)
            # print("model_langs:", model_langs)
            if data_langs==model_langs:
                model_list.append(model_path)
                
    print(f'no. of models being tested: {len(model_list)}')
    model_list.sort()
    for model_path in model_list:
        print(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = 'cuda'
        model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
        # model = BertAligner.from_pretrained(model_path, output_hidden_states=True).to(device)
        
        # model = torch.nn.DataParallel(model).to(device)

        evaluator = SquadEvaluator(tokenizer,
                                model,
                                load("squad_v2"),
                                )

        epoch = 0
        # eval on test
        epoch_test_loss = 0
        model.eval()
        split = 'test'
        progbar = tqdm(enumerate(dataset_test[split]),
                                total=len(dataset_test[split]),
                                desc=f"{split} - epoch {epoch + 1}")
        columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
        for i, batch in progbar:
            with torch.inference_mode():
                input = {k: batch[k].to('cuda') for k in columns}
                outputs = model(**input)
                # set to -10000 any logits in the query (left side of the inputs) so that the model cannot predict those tokens
                # for i in range(len(outputs['start_logits'])):
                #     outputs['start_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['start_logits'][i], input['token_type_ids'][i]-10000)
                #     outputs['end_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['end_logits'][i], input['token_type_ids'][i]-10000)
            # loss = outputs[0].mean()
            loss = outputs['loss'].mean()
            epoch_test_loss += loss.item()
            loss_tmp = round(epoch_test_loss / (i + 1), 4)
            progbar.set_postfix({'Loss': loss_tmp})
            
            evaluator.get_eval_batch(outputs, batch, split, type_labels = True)

        results_path = f'/home/pgajo/food/results/alignment/{model_langs}/test'
        # model save folder
        model_name_simple = model_path.split('/')[-1]
        data_name = args.data.split('/')[-1]
        data_results_path = os.path.join(results_path, data_name)
        if not os.path.isdir(data_results_path):
            os.makedirs(data_results_path)
        save_name = f"{model_name_simple}_E{evaluator.epoch_best}_{split.upper()}{str(round(evaluator.exact_dev_best, ndigits=0))}"

        metrics_save_path = os.path.join(results_path, data_name, save_name)
        if not os.path.exists(metrics_save_path):
            os.makedirs(metrics_save_path)
        
        evaluator.evaluate(model, split, epoch, eval_metric='test', model_name=model_name_simple, save_predictions=True)
        epoch_test_loss /= len(dataset_test[split])
        evaluator.epoch_metrics[f'{split}_loss'] = epoch_test_loss

        metrics = evaluator.print_metrics(current_epoch = epoch, current_split = split).to_dict()
        evaluator.store_metrics()

        metrics[0]['model_path'] = str(model_path)
        metrics[0]['data_test_path'] = str(args.data)
        

        with open(os.path.join(metrics_save_path, 'metrics.json'), 'w', encoding='utf8') as f:
            json.dump(metrics[0], f, ensure_ascii = False)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = argparse.ArgumentParser()
        parser.add_argument('-md', '--model_dir', help='Dir containing the models', default='./models')
        parser.add_argument('-d', '--data', help='Path of the test huggingface dataset', default="./datasets/alignment/en-it/DebertaV2TokenizerFast/GZ-GOLD_301_DebertaV2TokenizerFast_en-it")
        parser.add_argument('-t', '--types', default=False, action='store_true', help='Add a "_test" suffix to the repo name')
        args = parser.parse_args()
        
        main(args)