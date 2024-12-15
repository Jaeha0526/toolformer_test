import os
import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from prompts import retrieval_prompt
from data_generation.retrieval import RetrievalPostprocessing
from data_generation.calendar import CalendarPostprocessing
from data_generation.calculator import CalculatorPostprocessing
from data_generation.api_checker import check_apis_available
import json
import time
import argparse


def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do some continuations')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--num_devices", type=int, default=1)
    args = parser.parse_args()
    
    print("Initial GPU memory status:")
    print_gpu_memory()
    
    # Set GPU memory limits
    torch.cuda.set_per_process_memory_fraction(0.9)
    device = torch.device(f"cuda:{args.device_id}")
    
    # Initialize tokenizer
    gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    prompt_tokens = gpt_tokenizer(retrieval_prompt, return_tensors="pt")["input_ids"]
    start_tokens = [
        gpt_tokenizer("[")["input_ids"][0],
        gpt_tokenizer(" [")["input_ids"][0],
    ]
    end_tokens = [
        gpt_tokenizer("]")["input_ids"][0],
        gpt_tokenizer(" ]")["input_ids"][0],
    ]  # TODO: keep second?
    api_handler = CalculatorPostprocessing(start_tokens, end_tokens)
    
    # Add GPU memory management
    torch.cuda.empty_cache()
    
    print("After tokenizer initialization:")
    print_gpu_memory()
    
    # Load model in stages
    print("Loading model to CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print("Model loaded to CPU. Current GPU memory:")
    print_gpu_memory()
    
    # Enable memory saving features
    model.gradient_checkpointing_enable()
    
    # Move to GPU
    print("Moving model to GPU...")
    model = model.to(device)
    
    print("Model loaded to GPU. Current GPU memory:")
    print_gpu_memory()
    
    dataset = load_dataset("c4", "en", split="train", streaming=True, trust_remote_code=True)
    iter_data = iter(dataset)
    test = False
    counter = 0
    file_counter = 0
    found_examples = 0
    output_dataset = list()
    start_time = time.process_time()
    num_examples = int(25000.0/float(args.num_devices))
    start_count = -1
    if os.path.isfile(f"calc_data_{args.device_id}.json"):
        with open(f"calc_data_{args.device_id}.json") as f:
            output_dataset = json.load(f)
            start_count = output_dataset[-1]['file_index']
            for item in output_dataset:
                num_examples -= len(item['calculator_outputs'])
    while found_examples < num_examples:
        if found_examples % 10 == 0:
            print_gpu_memory()
        data = next(iter_data)
        if file_counter < start_count:
            file_counter += 1
            continue
        if file_counter % args.num_devices != args.device_id:
            file_counter += 1
            continue
        available = check_apis_available(data, gpt_tokenizer)
        test = available.calculator
        if test:
            # Clear GPU cache before processing each article
            torch.cuda.empty_cache()
            gc.collect()
            
            data_outputs = api_handler.parse_article(data, model, gpt_tokenizer)
            if len(data_outputs) == 0:
                eta_s = (num_examples - found_examples) * (time.process_time() - start_time) / max(1, found_examples)
                eta_m = eta_s // 60
                eta_h = eta_m // 60
                eta_m = eta_m - (eta_h * 60)
                eta_s = eta_s - ((eta_m * 60) + (eta_h * 60 * 60))
                print(f"device {args.device_id} Found: {found_examples}/{num_examples}, ETA: {eta_h}H:{eta_m}M:{eta_s}s")
                continue
            output_dataset.append(
                {
                    "file_index": file_counter,
                    "text": data["text"],
                    "calculator_outputs": data_outputs
                }
            )
            prev_found = found_examples
            found_examples += len(output_dataset[-1]["calculator_outputs"])
            eta_s = (num_examples - found_examples) * (time.process_time()-start_time) / max(1, found_examples)
            eta_m = eta_s // 60
            eta_h = eta_m // 60
            eta_m = eta_m - (eta_h*60)
            eta_s = eta_s - ((eta_m*60) + (eta_h*60*60))
            print(f"device {args.device_id} Found: {found_examples}/{num_examples}, ETA: {eta_h}H:{eta_m}M:{eta_s}s")
            if found_examples//100 > prev_found//100:
                with open(f"calc_data_{args.device_id}.json", 'w') as f:
                    json.dump(output_dataset, f, indent=2)
            counter += 1
            
            if found_examples % 10 == 0:  # More frequent memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
        file_counter += 1
    with open(f"calc_data_{args.device_id}.json", 'w') as f:
        json.dump(output_dataset, f, indent=2)