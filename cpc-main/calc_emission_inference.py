import argparse

# new
import logging
import subprocess
import sys
import time, os
from time import sleep
from tqdm import tqdm
import torch
from datasets import load_dataset
from pyJoules.energy_meter import measure_energy
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
)


# setup environment for energy measurements
def load_pipeline(model, tokenizer, task, device):
    pipe = pipeline(task=task, model=model, tokenizer=tokenizer, device=device)
    return pipe


# ----------------------------------------------------------------------------------------------------------------------
# sub stages of inference ( instead using pipeline() )
# breaking it down for greater control over individual operations


# takes raw input text data and prepares it for the model
def preprocessing_data(input, task, model_name, tokenizer, device):
    if task == "fill-mask":
        start = " ".join(input["target"].split()[:-1])
        suffix = " [MASK]."
        if (model_name == "roberta-base") or (model_name == "xlm-roberta-large"):
            suffix = " <mask>."
        processed_input = start + suffix
        input_tensor = tokenizer(processed_input, return_tensors="pt").to(device)
    else:
        # could the "and" lead to biased or incorrect results?
        processed_input = input["target"] + " and"  # why is "and" added here?
        input_tensor = tokenizer.encode(processed_input, return_tensors="pt").to(device)
    # print("processed_input: ",processed_input)
    return input_tensor


# runs the model using the preprocessed input data
# output depends on task (guessing vs. generating)
def perform_inference(model, task, inputs):
    with torch.no_grad():  # ensures no gradient computations are performed (not training)
        if task == "fill-mask":
            outputs = model(**inputs)
        else:
            outputs = model.generate(inputs, max_length=30, num_return_sequences=1)
    return outputs


# Extracts actual answer from model's output after inference
# raw output tensors w/probabilities --> text
# fill mask: highest probability predictions for masked tokens --> id's converted to text
# text gen: decodes sequence of token IDs into string
def extract_answer(tokenizer, task, outputs):
    if task == "fill-mask":
        # Get the predicted token IDs
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        # Convert token IDs back to text
        predicted_tokens = tokenizer.convert_ids_to_tokens(
            predicted_token_ids[0].tolist()
        )
        # Reconstruct the filled-in text
        result = tokenizer.convert_tokens_to_string(predicted_tokens)
    else:
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


# env is passed to energy measurements
# input_data is the text input reqd for processing
# model_name is used to handle specific conditions related to the specific task
# e.g. model_name decides the appropriate suffix for masked language modelling tasks
# general overarching measurement using HF pipeline
@measure_energy
def query(pipe, task, tokenizer, input_data, model_name):
    suffix = "."
    start = " ".join(input_data["target"].split()[:-1])
    if task == "fill-mask":
        suffix = " [MASK]."
        if (model_name == "roberta-base") or (model_name == "xlm-roberta-large"):
            suffix = " <mask>."
        processed_input = start + suffix
        curr_query = pipe(processed_input)
    else:
        processed_input = input_data["target"] + " and"
        curr_query = pipe(processed_input, max_length=30, num_return_sequences=1)
    # print("processed_input: ",processed_input)
    # print("answer: ", curr_query)
    return curr_query


# measures energy consumed by just the preprocessing step
@measure_energy
def query_just_data(tokenizer, task, input_data, device, model_name):
    inputs_pre = preprocessing_data(input_data, task, model_name, tokenizer, device)
    return inputs_pre


# measures energy at just inference
@measure_energy
def query_just_inference(task, inputs_pre, model):
    outputs_pre = perform_inference(model, task, inputs_pre)
    return outputs_pre


# measures energy at just the answer stage
@measure_energy
def query_just_extract(tokenizer, task, outputs_pre):
    answer = extract_answer(tokenizer, task, outputs_pre)
    return answer


# measures energy across inference and extract
# assumes data preprocessing is already done
@measure_energy
def query_just_inference_extract(tokenizer, task, inputs_pre, model):
    outputs_pre = perform_inference(model, task, inputs_pre)
    answer = extract_answer(tokenizer, task, outputs_pre)
    return answer


# measures energy across preprocessing, inference, and answer
@measure_energy
def query_2(tokenizer, task, input_data, model, device, model_name):
    inputs_pre = preprocessing_data(input_data, task, model_name, tokenizer, device)
    outputs_pre = perform_inference(model, task, inputs_pre)
    answer = extract_answer(tokenizer, task, outputs_pre)
    # print("answer: ",answer)
    return answer


# isolates energy measurements to get idle/baseline power usage?
@measure_energy
def tester_handler():
    sleep(3)
    return 0


# central point of execution
def main():

    # set some initial values for model and pipe <-- why?
    model = 0
    pipe = 0

    # ---------------- configurations: ----------------

    # defines and parses command-line arguments for the script p
    # args include -device, -task_type, -interface, and --test
    parser = argparse.ArgumentParser(
        prog="calc_emission",
        description="measure power consumption of GPU & CPU for NLP models",
        epilog="Text",
    )

    # user specifies:
    parser.add_argument("-device")  # GPU/CPU,
    parser.add_argument("-task_type")  # text-generation/fill-mask,
    parser.add_argument("-interface")  # how to run the task
    # (manually-just-inference, manually-just-inference-extract)
    parser.add_argument("--test", action="store_true")  # whether they're in test mode
    # action = "store_true" from argparse module for CLI args
    # this one is used with flag arg where standard setting is set to True
    # if you run the script like python script.py --test, then args.test will be True
    # if you run script without --test flag, then args.test will be False

    args = parser.parse_args()

    device = args.device
    task_type = args.task_type
    interface = args.interface

    # task_type options -> "text-generation" , "fill-mask"
    # device options -> "cuda" , "gpu", "cpu"
    # interface options -> "pipline" , "manually"

    # ---------------- outputs ----------------
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # check if error logs exist
    if not os.path.exists("error_logs"):
        os.makedirs("error_logs")
    if args.test:  # if in test mode, it prints to console
        if not os.path.exists("logs/"):
            os.makedirs("logs/")
        # check if error logs exist
        if not os.path.exists("error_logs/"):
            os.makedirs("error_logs/")
        print(
            "test mode, make sure output files are as expected before running script on models."
        )
        output_filename = f"logs/pyj_ot_{device}_{interface}_{task_type}.log"
        err_filename = "error_logs/pyj_ot_" + str(round(time.time())) + ".log"

    else:
        output_filename = f"logs/pyj_ot_{device}_{interface}_{task_type}.log"
        err_filename = f"error_logs/pyj_ot_{device}_{interface}_{task_type}.log"

    # opens files in write mode, creates if it doesn't already exist
    output_file = open(output_filename, "w")
    err_file = open(err_filename, "w")
    sys.stdout = output_file
    sys.stderr = err_file

    # Dataset loading to GPU
    # loads a validation split of dataset called common_gen from a collection gem
    dataset = load_dataset("gem", "common_gen", split="validation")

    # ---------------- models ----------------
    if task_type == "fill-mask":
        models_list = [
            "xlm-roberta-large",
            "bert-large-uncased",
            "roberta-base",
            "bert-base-uncased",
            "distilbert-base-uncased",
            "albert-base-v2",
        ]
    else:
        models_list = ["gpt2"]

    # subprocess to terminate processes and monitor the GPU
    # kills relevant processes that might interfere with energy measurement
    # subprocess.Popen("./nvkillprocess.sh", shell=True)
    if device == 'gpu' or device == 'cuda':
        subprocess.run(["pkill", "nvidia-smi"])

        # starts the logging of GPU utilization and power consumption detals
        # subprocess.Popen("./nvmodelprofile.sh", shell=True)  # place script here?
        # add three arguments to the script: device, task_type, interface
        subprocess.Popen("./nvmodelprofile.sh", device, task_type, interface, shell=True)

    if args.test:
        tester_handler()  # isolates measurements
        if device == 'gpu' or device == 'cuda':
            subprocess.Popen("./nvkillprocess.sh", shell=True)
        #else:
            # e.g. subprocess.ruh(["./cpu_cleanup.sh"]) 
        return 0

    # looping through models from the list
    for model_name in models_list:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ready to use pipeline with high-level abstraction provided by HF
        if interface == "pipeline":
            pipe = load_pipeline(
                task=task_type, model=model_name, tokenizer=tokenizer, device=device
            )

        # else check task type and load respective model using
        # AutoModelForMaskedLM or AutoModelForCasualLM depending on the task
        else:
            if task_type == "fill-mask":
                # moved to specific device (e.g. CPU or GPU)
                model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
            elif task_type == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            else:
                # if neither fill-mask or text-generation
                raise ValueError(f"Task type {task_type} is not supported.")

        # temp_input takes the 2nd entry from the loaded dataset <-- why?
        # import pdb; pdb.set_trace()
        temp_input = dataset[1]
        # preprocessing_data is used to process the input data
        # involves adding special tokens or modifying text, etc.
        first_input_data_proc = preprocessing_data(
            temp_input, task_type, model_name, tokenizer, device
        )
        if interface != "pipeline":
            first_outputs_pre = perform_inference(
                model, task_type, first_input_data_proc
            )

        # ---------------- experiment ----------------
        # why is err_msg used for normal logging operations?
        err_msg = f"current_model: {model_name}, timestamp: {round(time.time())}\n"
        sys.stderr.write(err_msg)
        sleep(2)

        for _i in tqdm(range(2)): # changed from 10 to 2
            if interface == "manually-just-inference":
                query_just_inference(task_type, first_input_data_proc, model)

            elif interface == "manually-just-inference-extract":
                query_just_inference_extract(
                    tokenizer, task_type, first_input_data_proc, model
                )

            elif interface == "manually-just-extract":
                query_just_extract(tokenizer, task_type, first_outputs_pre)

            else:  # iterates through the dataset
                for input_data in dataset:

                    if (
                        interface == "manually"
                    ):  # calls query 2: processing + inference + answer extraction
                        query_2(
                            tokenizer, task_type, input_data, model, device, model_name
                        )

                    elif interface == "manually-just-data":
                        query_just_data(
                            tokenizer, task_type, input_data, device, model_name
                        )

                    else:
                        query(pipe, task_type, tokenizer, input_data, model_name)

    err_msg = "end experiment :)"
    sys.stderr.write(err_msg)
    #subprocess.Popen("./nvkillprocess.sh", shell=True)


if __name__ == "__main__":
    main()
