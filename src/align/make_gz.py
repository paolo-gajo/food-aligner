import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import TASTEset
from transformers import AutoTokenizer
import warnings
import os

def main():
    unshuffled_size = 1
    shuffled_size = 0

    tokenizer_name = 'bert-base-multilingual-cased'
    # tokenizer_name = 'microsoft/mdeberta-v3-base'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data_path = './data/GZ/GZ_GOLD/GZ-GOLD_301.json'
    # data_path = './data/mycolombianrecipes/MCR-GOLD_291.json'
    class_filter = [
        # 'FOOD',
        # 'QUANTITY',
        # 'UNIT',
        # 'PROCESS',
        # 'PHYSICAL_QUALITY',
        # 'COLOR',
        # 'TASTE',
        # 'PURPOSE',
        # 'PART',
    ]
    all_classes = ['FOOD', 'QUANTITY', 'UNIT', 'PROCESS', 'PHYSICAL_QUALITY', 'COLOR', 'TASTE', 'PURPOSE', 'PART',]
    src_lang = 'en'
    tgt_langs = ['it']
    dataset = TASTEset.from_json(
        data_path,
        tokenizer = tokenizer,
        src_lang = src_lang,
        tgt_langs = tgt_langs, # N.B.: the actual source language in GZ is italian, but our models were trained on english to predict an italian target
        dev_size = 0,
        shuffled_size = shuffled_size,
        unshuffled_size = unshuffled_size,
        # aligned = False,
        # debug_dump = True,
        drop_duplicates = False,
        label_studio = True,
        # inverse_languages = True,
        # verbose = True,
        # n_rows = 301
        class_filter=class_filter,      
        text_field = 'ingredients'
        )

    tgt_langs_string = ''.join(dataset.tgt_langs)
    save_name = f"{data_path.split('/')[-1].replace('.json', '')}_{type(tokenizer).__name__}_{dataset.src_lang}-{tgt_langs_string}{'_' + '-'.join(list(set(all_classes).difference(set(class_filter)))) if len(class_filter)>0 else ''}"
    repo_id = f"pgajo/{save_name}"
    print('repo_id:', repo_id)
    # dataset.push_to_hub(repo_id)
    # dataset_summary = f'''
    # Tokenizer: {type(tokenizer).__name__}\n
    # Dataset: {dataset.name}\n
    # Unshuffled ratio: {dataset.unshuffled_size}\n
    # Shuffled ratio: {dataset.shuffled_size}\n
    # Shuffle probability: {dataset.shuffle_probability}\n
    # Drop duplicates: {dataset.drop_duplicates}\n
    # Dataset path = {dataset.data_path}\n
    # '''
    # push_dataset_card(repo_id, dataset_summary=dataset_summary)
    datasets_dir_path = f"/home/pgajo/food/datasets/alignment/{dataset.src_lang}-{tgt_langs_string}/{type(tokenizer).__name__}/"
    if not os.path.exists(datasets_dir_path):
        os.makedirs(datasets_dir_path)
    dataset.save_to_disk(os.path.join(datasets_dir_path, save_name))


    # dataset.name = data_path.split('/')[-1].replace('.json', '')
    # save_name = f"{type(tokenizer).__name__}_{dataset.name}_U{dataset.unshuffled_size}_S{dataset.shuffled_size}_DROP{str(int(dataset.drop_duplicates))}"
    # repo_id = f"pgajo/{save_name}_types"
    # print('repo_id:', repo_id)
    # local_dir = data_path.replace('.json', '')
    # if not os.path.isdir(local_dir):
    #     os.makedirs(local_dir)
    # # dataset.save_to_disk(local_dir)
    # dataset.push_to_hub(repo_id)
    # dataset_summary = f'''
    # Tokenizer: {type(tokenizer).__name__}\n
    # Dataset: {dataset.name}\n
    # Unshuffled ratio: {dataset.unshuffled_size}\n
    # Shuffled ratio: {dataset.shuffled_size}\n
    # Drop duplicates: {dataset.drop_duplicates}\n
    # Dataset path = {dataset.data_path}\n
    # '''
    # push_dataset_card(repo_id, dataset_summary=dataset_summary, model_metrics = '')

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()