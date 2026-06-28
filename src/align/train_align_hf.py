import warnings
import torch
torch.set_printoptions(linewidth=100000, threshold=100000)
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering, BertModel, PretrainedConfig, BertPreTrainedModel
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import data_loader, SquadEvaluator, TASTEset, save_local_model
from aligner_pt import BertAligner
import re
from datetime import datetime
import os
import argparse

def main(args):
    data_name_simple = args.data_name.split('/')[-1]

    data = TASTEset.from_disk(args.data_name)

    batch_size = 16
    dataset = data_loader(data,
                        batch_size,
                        # n_rows=320,
                        )
    device = 'cuda'
    bertaligner = 0
    if bertaligner:
        model = BertAligner.from_pretrained(args.model_name,
                                        output_hidden_states=True).to(device)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name).to(device)
    # model = BertCrf(2, args.model_name)
    
    # model = torch.nn.DataParallel(model).to(device)
    
    lr = 3e-5
    eps = 1e-8
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                lr = lr,
                                eps = eps
                                )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    evaluator = SquadEvaluator(tokenizer,
                            model,
                            load("squad_v2"),
                            )
    bottom_out_non_context = 0
    epochs = 3
    for epoch in range(epochs):
        # train
        epoch_train_loss = 0
        model.train()
        split = 'train'
        progbar = tqdm(enumerate(dataset[split]),
                                total=len(dataset[split]),
                                desc=f"{split} - epoch {epoch + 1}")
        print('args.model_name:', args.model_name)
        print('args.data_name:', args.data_name)
        columns = [
                    'input_ids',
                    'token_type_ids',
                    'attention_mask',
                    'start_positions',
                    'end_positions'
                    ]
        for i, batch in progbar:
            input = {k: batch[k].to(device) for k in columns}
            outputs = model(**input) # ['loss', 'start_logits', 'end_logits']
            if bottom_out_non_context:
                for i in range(len(outputs['start_logits'])):
                    outputs['start_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['start_logits'][i], -10000)
                    outputs['end_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['end_logits'][i], -10000)
            
            loss = outputs['loss']#.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            loss_tmp = round(epoch_train_loss / (i + 1), 4)
            progbar.set_postfix({'Loss': loss_tmp})
            
            evaluator.get_eval_batch(outputs, batch, split)

        evaluator.evaluate(model, split, epoch)
        epoch_train_loss /= len(dataset[split])
        evaluator.epoch_metrics[f'{split}_loss'] = epoch_train_loss

        evaluator.print_metrics(current_epoch = epoch, current_split = split)

        # eval on dev
        epoch_dev_loss = 0
        model.eval()
        split = 'dev'
        progbar = tqdm(enumerate(dataset[split]),
                                total=len(dataset[split]),
                                desc=f"{split} - epoch {epoch + 1}")
        print('args.model_name:', args.model_name)
        print('args.data_name:', args.data_name)
        for i, batch in progbar:
            input = {k: batch[k].to(device) for k in columns}
            with torch.inference_mode():
                outputs = model(**input)
            loss = outputs['loss']#.mean()
            epoch_dev_loss += loss.item()
            loss_tmp = round(epoch_dev_loss / (i + 1), 4)
            progbar.set_postfix({'Loss': loss_tmp})
            
            evaluator.get_eval_batch(outputs, batch, split)
        
        evaluator.evaluate(model, split, epoch)
        epoch_dev_loss /= len(dataset[split])
        evaluator.epoch_metrics[f'{split}_loss'] = epoch_dev_loss

        evaluator.print_metrics(current_epoch = epoch, current_split = split)

        evaluator.store_metrics()

        if evaluator.stop_training:
            print(f'Early stopping triggered on epoch {epoch}. \
                \nBest epoch: {evaluator.epoch_best}.')                                               
            break
    
    evaluator.print_metrics()

    if bertaligner:
        args.model_name = f'{args.model_name}_BertAligner'
    else:
        args.model_name = args.model_name.split('/')[-1]
    results_path = f'/home/pgajo/food/results/alignment/{args.model_name}'
    model_dir = f'/home/pgajo/food/models/alignment/{args.model_name}'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    args.data_name = re.sub('.json', '', args.data_name.split('/')[-1]) # remove extension if local path
    data_results_path = os.path.join(results_path, args.data_name)
    if not os.path.isdir(data_results_path):
        os.makedirs(data_results_path)
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_name = f"E{evaluator.epoch_best}_DEV{str(round(evaluator.exact_dev_best, ndigits=0))}_ME{epochs}_{date_time}"
    save_name = save_name.replace('bert-base-multilingual-cased', 'mbert')
    save_name = save_name.replace('mdeberta-v3-base', 'mdeberta')
    csv_save_path = os.path.join(data_results_path, save_name)
    print('Saving metrics to:', csv_save_path)
    evaluator.save_metrics_to_csv(csv_save_path)

    model_save_dir = os.path.join(model_dir, f"{args.model_name}_{data_name_simple}_ME{epochs}_{date_time}")
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    evaluator.save_metrics_to_csv(os.path.join(model_save_dir, 'metrics'))
    save_local_model(model_save_dir, model, tokenizer)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = argparse.ArgumentParser(description="A sample argparse program")
        parser.add_argument("--model_name", help="local/huggingface path of the model to train", default = 'microsoft/mdeberta-v3-base')
        parser.add_argument("--data_name", help="path of the dataset to train on", default = 'datasets/alignment/en-it/DebertaV2TokenizerFast/EW-TT-MT_multi_ctx_P0.3_en-it')
        args = parser.parse_args()
        # args.model_name = 'microsoft/mdeberta-v3-base'
        # args.model_name = 'google-bert/bert-base-multilingual-cased'
        
        # args.data_name = 'datasets/alignment/en-it/DebertaV2TokenizerFast/EW-TT-MT_multi_ctx_P0.3_en-it'
        # args.data_name = './datasets/alignment/en-it/DebertaV2TokenizerFast/mdeberta-v3-base/mdeberta_xlwa_en-it'
        # args.data_name = './datasets/alignment/en-it/BertTokenizerFast/bert-base-multilingual-cased/mbert_xlwa_en-it'
        main(args)
