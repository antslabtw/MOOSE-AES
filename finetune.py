import fire
from math import ceil
import os
import torch
import pickle
from transformers import TrainingArguments
from torch.utils.data import RandomSampler, DataLoader
from data.prompt_dataloader import CustomDataset
from model.multi_bert_moose import multiBert as Model
from trainers import BertTrainer
from utils.callbacks import EvaluateRecord
from utils.general_utils import seed_all
from utils.multitask_evaluator_all_attributes import Evaluator
from safetensors.torch import load_file
from torch.utils.data.dataloader import default_collate


class NerConfig:
    def __init__(self):
        self.lr = 1e-4
        self.epoch = 15
        self.batch_size = 8
        self.device = "cuda"
        self.num_trait = 9
        self.alpha = 0.7
        self.delta = 0.7
        self.filter_num = 256
        self.chunk_sizes = [90, 30, 130, 10]
        self.hidden_dim = 256 # chunk & linear output_dim
        self.mhd_head = 2


args = NerConfig()


def train(
    test_prompt_id: int = 1,
    experiment_tag: str = "test",
    seed: int = 11,
    num_train_epochs: int = 14,
    batch_size: int = 8,
    gradient_accumulation: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.001,
    chunk_sizes: int = [90, 30, 130, 10],
    tag: str = "trait"
):
    seed_all(seed)

    train_dataset = CustomDataset(f"/feacture/var_norm/new_train/encode_prompt_{test_prompt_id}.pkl")
    eval_dataset = CustomDataset(f"/feacture/var_norm/new_dev/encode_prompt_{test_prompt_id}.pkl")
    test_dataset = CustomDataset(f"/feacture/var_norm/new_test/encode_prompt_{test_prompt_id}.pkl")
    model = Model(
        args=args
    )
    evaluator = Evaluator(eval_dataset, test_dataset, seed)

    output_dir = f"ckpts/{tag}/meanvar_base_kv_prompt_{test_prompt_id}"
    logging_dir = f"logs/{experiment_tag}/ckpts/{tag}/meanvar_base_kv_prompt_{test_prompt_id}"
    
    num_devices = 1
    N = len(train_dataset) 
    B = batch_size * num_devices
    G = gradient_accumulation
    steps_per_epoch = N / (B * num_devices * G)
    steps_per_quarter_epoch = int(steps_per_epoch * 0.25)


    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate = learning_rate,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation,
        logging_dir = logging_dir,
        # eval_strategy = "epoch",
        eval_strategy="steps",
        eval_steps=steps_per_quarter_epoch,
        label_names = ["scaled_score"],
        save_strategy = "epoch",
        save_total_limit = 20,
        do_eval = True,
        load_best_model_at_end = False, 
        fp16 = True,
        remove_unused_columns = True,
        metric_for_best_model = "eval_test_avg",
        greater_is_better = True,
        seed = seed,
        data_seed = seed,
        ddp_find_unused_parameters = False,
        weight_decay = weight_decay
    )
            
    trainer = BertTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        test_dataset = test_dataset,
        evaluator = evaluator,
        callbacks = [EvaluateRecord(output_dir)],
        data_collator = default_collate,
    )

    print('Trainer is using device:', trainer.args.device)
    print(test_prompt_id)
    trainer.train()

if __name__ == "__main__":
    # import sys
    # train(test_prompt_id = int(sys.argv[1]))
    fire.Fire(train)