import warnings
import argparse
import os
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import TASTEset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='dummy argument to make the script work on Google Colab')
    parser.add_argument('-i', '--input', default='/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_multi_ctx.json', help='path of the input json dataset')
    parser.add_argument('-o', '--output', default='', help='path of the input json dataset')
    parser.add_argument('-l', '--shuffle_languages', default='it', help='space-separated 2-character codes of the dataset target languages to shuffle')
    parser.add_argument('-src', '--src_lang', default='en', help='space-separated 2-character code of the dataset source language')
    parser.add_argument('-t', '--tokenizer_name', default='bert-base-multilingual-cased', help='tokenizer to use')
    parser.add_argument('-d', '--drop_duplicates', default=True, help='if True (default=True), drop rows with the same answer')
    parser.add_argument('-ss', '--shuffled_size', default=1, help='length multiplier for the number of shuffled instances (default=1)')
    parser.add_argument('-us', '--unshuffled_size', default=1, help='length multiplier for the number of unshuffled instances (default=1)')
    parser.add_argument('-ds', '--dev_size', default=1, help='size of the dev split (default=0.2)')

    args = parser.parse_args()

    args.unshuffled_size = 0
    args.shuffled_size = 1
    args.drop_duplicates = True

    tokenizer_name = 'google-bert/bert-base-multilingual-cased'
    # tokenizer_name = 'microsoft/mdeberta-v3-base'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    args.input = './data/TASTEset/data/EW-TT-MT_multi_ctx.json'

    langs = [
        'it',
        # 'es',
        # 'de',
    ]
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
    
    for i in range(3,11):
        # if i > 2:
        #     break
        dataset = TASTEset.from_json(
            args.input,
            tokenizer = tokenizer,
            tgt_langs = langs,
            src_lang = 'en',
            dev_size = 0.2,
            shuffled_size = args.shuffled_size,
            unshuffled_size = args.unshuffled_size,
            shuffle_type = 'ingredient',
            shuffle_probability = i/10,
            drop_duplicates = 0,
            class_filter = class_filter
            )

        save_name = f"{args.input.split('/')[-1].replace('.json', '')}_P{dataset.shuffle_probability}_en-{'-'.join(langs)}{'_' + '-'.join(list(set(all_classes).difference(set(class_filter)))) if len(class_filter)>0 else ''}"
        repo_id = f"pgajo/{save_name}"
        print('repo_id:', repo_id)
        # dataset.push_to_hub(repo_id)
        # dataset_summary = f'''
        # Tokenizer: {tokenizer_dict[tokenizer_name]}\n
        # Dataset: {dataset.name}\n
        # Unshuffled ratio: {dataset.unshuffled_size}\n
        # Shuffled ratio: {dataset.shuffled_size}\n
        # Shuffle probability: {dataset.shuffle_probability}\n
        # Drop duplicates: {dataset.drop_duplicates}\n
        # Dataset path = {dataset.data_path}\n
        # '''
        # push_dataset_card(repo_id, dataset_summary=dataset_summary)
        datasets_dir_path = f"/home/pgajo/food/datasets/alignment/en-{'-'.join(langs)}/{str(type(tokenizer).__name__)}"
        if not os.path.exists(datasets_dir_path):
            os.makedirs(datasets_dir_path)
        full_save_path = os.path.join(datasets_dir_path, save_name)
        print(full_save_path)
        dataset.save_to_disk(full_save_path)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()