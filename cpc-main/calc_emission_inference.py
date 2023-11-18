
import argparse
import subprocess
import sys
import torch
import time
from datetime import datetime
from time import sleep

from huggingface_hub.utils import tqdm
from transformers import pipeline, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from pyJoules.energy_meter import measure_energy
from pyJoules.device.rapl_device import RaplCoreDomain

# pipeline interface

def load_pipeline(model, tokenizer, task, device):
    pipe = pipeline(task=task, model=model, tokenizer=tokenizer, device=device)
    return pipe


@measure_energy
def query(pipe, task, tokenizer, input_data, model_name):
    suffix = '.'
    start = ' '.join(input_data['target'].split()[:-1])
    if task == "fill-mask":
        suffix = ' [MASK].'
        if (model_name == "roberta-base") or (model_name == "xlm-roberta-large"):
            suffix = ' <mask>.'
        processed_input = start + suffix
        curr_query = pipe(processed_input)
    else:
        processed_input = input_data['target'] + " and"
        curr_query = pipe(processed_input, max_length=30, num_return_sequences=1)
    # print("processed_input: ",processed_input)
    # print("answer: ", curr_query)
    return curr_query


# ----------------------------------------------------------------------------------------------------------------------
# sub stages of inference ( instead using pipeline() )

def preprocessing_data(input, task, model_name, tokenizer, device):
    if task == "fill-mask":
        start = ' '.join(input['target'].split()[:-1])
        suffix = ' [MASK].'
        if (model_name == "roberta-base") or (model_name == "xlm-roberta-large"):
            suffix = ' <mask>.'
        processed_input = start + suffix
        input_tensor = tokenizer(processed_input, return_tensors="pt").to(device)
    else:
        processed_input = input['target'] + " and"
        input_tensor = tokenizer.encode(processed_input, return_tensors="pt").to(device)
    # print("processed_input: ",processed_input)
    return input_tensor


def perform_inference(model, task, inputs):
    # no grad because only for inference
    with torch.no_grad():
        if task == "fill-mask":
            outputs = model(**inputs)
        else:
            outputs = model.generate(inputs, max_length=30, num_return_sequences=1)
    return outputs


def extract_answer(tokenizer, task, outputs):
    if task == "fill-mask":
        # Get the predicted token IDs
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        # Convert token IDs back to text
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids[0].tolist())
        # Reconstruct the filled-in text
        result = tokenizer.convert_tokens_to_string(predicted_tokens)
    else:
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


@measure_energy
def query_2(tokenizer, task, input_data, model, device, model_name):
    inputs_pre = preprocessing_data(input_data, task, model_name, tokenizer, device)
    outputs_pre = perform_inference(model, task, inputs_pre)
    answer = extract_answer(tokenizer, task, outputs_pre)
    # print("answer: ",answer)
    return answer


@measure_energy
def query_just_inference_extract(tokenizer, task, inputs_pre, model):
    outputs_pre = perform_inference(model, task, inputs_pre)
    answer = extract_answer(tokenizer, task, outputs_pre)
    return answer
@measure_energy
def query_just_inference(task, inputs_pre, model):
    outputs_pre = perform_inference(model, task, inputs_pre)
    return outputs_pre

@measure_energy
def query_just_data(tokenizer, task, input_data, device, model_name):
    inputs_pre = preprocessing_data(input_data, task, model_name, tokenizer, device)
    return inputs_pre



@measure_energy
def tester_handler():
    sleep(3)
    return 0


def main():
    model = 0
    pipe = 0
    # ---------------- configurations: ----------------
    parser = argparse.ArgumentParser(
        prog='calc_emission',
        description='measure power consumption of GPU & cPU for NLP models',
        epilog='Text')

    parser.add_argument('-device')
    parser.add_argument('-task_type')
    parser.add_argument('-interface')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    device = args.device
    task_type = args.task_type
    interface = args.interface

    # task_type options -> "text-generation" , "fill-mask"
    # device options -> "cuda" , "gpu", "cpu"
    # interface options -> "pipline" , "manually"

    # ---------------- outputs ----------------
    if args.test:
        print("test mode, make sure output files are as expected before running script on models ")
        output_filename = "pyjoules_output_test_" + str(round(time.time())) + ".log"
        err_filename = "err_test_" + str(round(time.time())) + ".log"
    else:
        output_filename = "pyjoules_output_" + task_type + "_" + interface + "_" + device + "_" + str(
            round(time.time())) + ".log"
        err_filename = "err_" + task_type + "_" + interface + "_" + device + "_" + str(
            round(time.time())) + ".log"
    output_file = open(output_filename, "w")
    err_file = open(err_filename, "w")
    sys.stdout = output_file
    sys.stderr = err_file
    # Dataset loading to GPU
    dataset = load_dataset("gem", "common_gen", split="validation")

    # ---------------- models ----------------
    if task_type == "fill-mask":
        models_list = ["xlm-roberta-large", "bert-large-uncased", "roberta-base", "bert-base-uncased",
                       "distilbert-base-uncased", "albert-base-v2"]
    else:
        models_list = ["gpt2"]

    subprocess.Popen("./nvkillprocess.sh", shell=True)
    subprocess.Popen("./nvmodelprofile.sh", shell=True)

    if args.test:
        tester_handler()
        subprocess.Popen("./nvkillprocess.sh", shell=True)
        return 0

    for model_name in models_list:
        # print("\n ------------------------------ current_model:", model_name, " ------------------------------\n")
        # Model loading
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if interface == "pipeline":
            pipe = load_pipeline(task=task_type, model=model_name, tokenizer=tokenizer, device=device)
        else:
            if task_type == "fill-mask":
                model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
            elif task_type == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            else:
                raise ValueError(f"Task type {task_type} is not supported.")

        temp_input = dataset[1]
        first_input_data_proc = preprocessing_data(temp_input, task_type, model_name, tokenizer, device)

        # ---------------- experiment ----------------
        err_msg = "current_model: " + model_name + ", timestamp: " + str(round(time.time())) +"\n"
        sys.stderr.write(err_msg)
        sleep(2)
        for i in range(10):
            if interface == "manually-just-inference":
                query_just_inference(task_type, first_input_data_proc, model)
            elif interface == "manually-just-inference-extract":
                query_just_inference_extract(tokenizer, task_type, first_input_data_proc, model)
            else:
                for input_data in dataset:
                    if interface == "manually":
                        query_2(tokenizer, task_type, input_data, model, device, model_name)
                    elif interface == "manually-just-data":
                        #sleep(0.01)
                        query_just_data(tokenizer, task_type, input_data, device, model_name)
                    else:
                        query(pipe, task_type, tokenizer, input_data, model_name)

    err_msg = "end experiment :)"
    sys.stderr.write(err_msg)
    subprocess.Popen("./nvkillprocess.sh", shell=True)



if __name__ == "__main__":
    main()
