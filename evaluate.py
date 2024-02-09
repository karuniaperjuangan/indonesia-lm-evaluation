import argparse
import sys
import pandas as pd
import math
import os
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,BitsAndBytesConfig
from tqdm import tqdm
from numpy import argmax
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import yaml
import re
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


def prepare_data_from_yaml(yaml_path):
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
        example_idxs = np.random.default_rng(seed=456368).integers(0, len(inputs), size=config['num_shot'])
        shot_examples = []
        for idx in range(config['num_shot']):
            shot_examples.append(inputs[example_idxs[idx]] + labels[example_idxs[idx]])
        shot_examples = "\n\n".join(shot_examples) + "\n\n"
        for idx in range(len(inputs)):
            inputs[idx] = shot_examples + inputs[idx]
    return inputs, labels, outputs_options


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_8bit", action='store_true')
    parser.add_argument("--load_4bit", action='store_true')
    parser.add_argument("--share_gradio", action='store_true')
    parser.add_argument("--by_letter", action='store_true')
    parser.add_argument("--base_model", type=str, help="Path to pretrained model", required=True)
    parser.add_argument("--lora_weights", type=str, default="x")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--output_folder", type=str, default="./output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    tokenizer_class = LlamaTokenizer if 'llama' in args.base_model else AutoTokenizer
    model_class = LlamaForCausalLM if 'llama' in args.base_model else AutoModelForCausalLM

    SAVE_FILE = f'{args.output_folder}/result_{args.base_model.split("/")[-1]}_{args.by_letter}.csv'
    tokenizer = tokenizer_class.from_pretrained(args.base_model)
    
    if 'mt0' in args.base_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, device_map="auto", load_in_8bit="xxl" in args.base_model)
        from utils import predict_classification_mt0 as predict_classification
        from utils import predict_classification_mt0_by_letter as predict_classification_by_letter
    else:
        if args.load_8bit and args.load_4bit:
            load_in_8bit = False
        else:
            load_in_8bit = args.load_8bit
        model = model_class.from_pretrained(args.base_model,
                                            quantization_config=nf4_config if args.load_4bit else None,
                                            load_in_8bit=load_in_8bit,
                                            torch_dtype=torch.float16,
                                            trust_remote_code=True, 
                                            device_map="auto",)

        from utils import predict_classification_causal as predict_classification
        from utils import predict_classification_causal_by_letter as predict_classification_by_letter
    from utils import predict_classification_causal_by_letter_new as predict_classification_by_letter_new
    
    # Load adapter if we use adapter
    if args.lora_weights != "x":
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            torch_dtype=torch.float16,
        )
        SAVE_FILE = f'{args.output_folder}/result_{args.base_model.split("/")[-1]}_{args.lora_weight.split("/")[-1]}_{args.by_letter}.csv'

    # unwind broken decapoda-research config
    if 'llama' in args.base_model:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    model.eval()
    #if torch.__version__ >= "2" and sys.platform != "win32":
    #    model = torch.compile(model)
    
    
    inputs, labels, outputs_options = prepare_data_from_yaml(args.config_path)
    preds = []
    probs = []
    for idx in tqdm(range(len(inputs))):
        prob, pred = predict_classification_by_letter_new(model, tokenizer, inputs[idx], outputs_options[idx], device)
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
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc}")
    output_df = pd.DataFrame()
    output_df['input'] = inputs
    output_df['label'] = labels
    output_df['options'] = outputs_options
    output_df['preds'] = preds
    output_df['probs'] = probs
    output_df.to_csv(SAVE_FILE, index=False)

if __name__ == "__main__":
    main()
