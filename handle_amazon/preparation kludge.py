import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk, get_dataset_config_names

from copy import deepcopy
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
from functools import partial


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def tokenization(example):
    return tokenizer(example["review_body"], truncation=True, padding=True, pad_to_multiple_of=512, max_length=512)

def tokenization_en(example):
    return tokenizer(example["en_review_body"], truncation=True, padding=True, pad_to_multiple_of=512, max_length=512)

def compute_bin_label(example, max_neg=2, min_pos=4):
    example["bin_label"] = 1 if example["stars"] >= min_pos else (0 if example["stars"] <= max_neg else -1)
    return example

translators = {
    lang: pipeline("translation", model=f"Helsinki-NLP/opus-mt-{lang}-en", batch_size=8, device='cpu', max_length=150)
    for lang in ['fr', 'de', 'es']
}
def compute_translation(example, lang=None):
    example['en_review_body'] = [x['translation_text'] for x in translators[lang]([x[:2000] for x in example['review_body']])]
    return example

def id_translation(example):
    example['en_review_body'] = example['review_body']
    return example

def normalize_dataset(dataset):
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "en_input_ids", "en_attention_mask", "stars", "bin_label"])


def main():
    subsets = ['train', 'validation', 'test']
    config_names = get_dataset_config_names("amazon_reviews_multi")
    print(config_names)
    # multi_amazon = load_dataset("amazon_reviews_multi")

    lang_list = ['en', 'fr', 'de', 'es']
    data = {
        lang: load_from_disk(f'amazon_{lang}')
        for lang in lang_list
    }

    tr_data = {
        lang: load_from_disk(f'amazon_tr_{lang}')
        for lang in lang_list
    }


    for lang in ['de', 'en', 'es', 'fr']:
        ldata = tr_data[lang]
        ldata['train'] = data[lang]['train'].map(id_translation, batched=True)

        ldata['train'] = ldata['train'].map(tokenization_en, batched=True)
        ldata['train'] = ldata['train'].rename_column('input_ids', 'en_input_ids')
        ldata['train'] = ldata['train'].rename_column('attention_mask', 'en_attention_mask')
        ldata['train'] = ldata['train'].map(tokenization, batched=True)

        for subset in subsets:
            normalize_dataset(ldata[subset])
        ldata.save_to_disk(f'amazon_ok_tr_{lang}')
        print(ldata)

if __name__ == "__main__":
    main()