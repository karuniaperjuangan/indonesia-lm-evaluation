import argparse
import sys
import pandas as pd
import math
import os
from tqdm import tqdm
from numpy import argmax
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import yaml
import re
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"




def prepare_data_from_yaml(yaml_path, batch_size=1):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data = pd.read_csv(config['data_path'])
    inputs = []
    labels = []
    outputs_options = []
    template_prompt = config['prompt']
    # find all [KEY] in prompt
    keys = re.findall(r'(\[[A-Z]+\])', template_prompt)
    for idx, row in data.iterrows():
        prompt = template_prompt
        for key in keys:
            prompt = prompt.replace(key, row[key])
        inputs.append(prompt)
        labels.append(row[config['label_column']])
        outputs_options.append(config['options'])
    if config['num_shot'] > 0:
        if config['num_shot'] <= len(config['options']):
            i =0
            shot_examples = []
            for option in config['options']:
                if i >= config['num_shot']:
                    break
                i += 1
                #index with label column containing the option
                example_idxs = data[data[config['label_column']] == option].sample(1,random_state=456367).index
                shot_examples.append(inputs[example_idxs[0]] + labels[example_idxs[0]])
                
            shot_examples = "\n\n".join(shot_examples) + "\n\n"
            for idx in range(len(inputs)):
                inputs[idx] = shot_examples + inputs[idx]
            

        else:
            example_idxs = np.random.default_rng(seed=456368).integers(0, len(inputs), size=config['num_shot'])
            shot_examples = []
            for idx in range(config['num_shot']):
                shot_examples.append(inputs[example_idxs[idx]] + labels[example_idxs[idx]])
            shot_examples = "\n\n".join(shot_examples) + "\n\n"
            for idx in range(len(inputs)):
                inputs[idx] = shot_examples + inputs[idx]
    inputs = [config.get('instruction', '') + inp for inp in inputs]
    #print(inputs[0])
    #print(labels[0])
    #print(outputs_options[0])
    #raise Exception("stop")
    #batching
    #inputs = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
    #labels = [labels[i:i+batch_size] for i in range(0, len(labels), batch_size)]
    #outputs_options = [outputs_options[i:i+batch_size] for i in range(0, len(outputs_options), batch_size)]
    #print(f"Number of batches: {len(inputs)}")
    return inputs, labels, outputs_options


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_8bit", action='store_true')
    parser.add_argument("--load_4bit", action='store_true')
    parser.add_argument("--share_gradio", action='store_true')
    parser.add_argument("--by_letter", action='store_true')
    parser.add_argument("--base_model", type=str, help="Path to pretrained model", required=True)
    parser.add_argument("--lora_path", type=str, help="Path to pretrained lora model")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--output_folder", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    model_directory = args.base_model
    print("Loading model: " + model_directory)

    config = ExLlamaV2Config(model_directory)
    SAVE_FILE = f'{args.output_folder}/result_{args.base_model.split("/")[-1]}_{args.by_letter}.csv'
    tokenizer = ExLlamaV2Tokenizer(config)
    tokenizer.padding_side = "left" # padding must be on the beginning since we predict from the end
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    model = ExLlamaV2(config)
    model.load()
    if args.lora_path:
        print("Loading LORA model: " + args.lora_path)
        lora = [ExLlamaV2Lora.from_directory(model, args.lora_path)]
    else:
        lora = []

    from exllama_utils import predict_classification_causal_single_token, predict_classification_multi_token

    
    
    inputs, labels, outputs_options = prepare_data_from_yaml(args.config_path, batch_size=args.batch_size)
    preds = []
    probs = []
    for idx in tqdm(range(len(inputs))):
        prob, pred = predict_classification_causal_single_token(model, tokenizer, inputs[idx], outputs_options[idx],loras=lora)
        preds.append(pred)
        probs.append(prob)
        """
        if not args.by_letter:
            out = predict_classification(model, tokenizer, inputs[idx], outputs_options[idx], device)
            prob = [o.cpu().detach().item() for o in out]
            pred = argmax(prob)
            preds.append(pred)
            probs.append(prob)
        else:
            conf, pred = predict_classification_by_letter(model, tokenizer, inputs[idx], outputs_options[idx], device)
            probs.append(conf)
            preds.append(pred)
        """
    
    output_df = pd.DataFrame()
    # unpack the batches
    output_df['input'] = inputs
    output_df['label'] = labels
    output_df['options'] = outputs_options
    output_df['preds'] = preds
    output_df['probs'] = probs
    output_df.to_csv(SAVE_FILE, index=False)

    acc = accuracy_score([label for batch in labels for label in batch], [pred for batch in preds for pred in batch])
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    main()
