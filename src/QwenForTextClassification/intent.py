import ast
import evaluate
import numpy as np
import pandas as pd
import torch
from config.dataset import DataArguments, InferArguments
from config.model_args import ModelArguments, NLUTrainingArguments
from datasets import Dataset
from model.qwen import Qwen2ForSequenceClassification
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    pipeline,
)
from sklearn.metrics import precision_score, recall_score, f1_score


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset_from_jsonl(file_path, intent_dict):
    df = pd.read_json(file_path, lines=True)
    df["intent"] = df["intent"].apply(lambda x: intent_dict[x])
    dataset_dict = {
        "question": df["question"].tolist(),
        "intent": df["intent"].tolist(),
    }
    return Dataset.from_dict(dataset_dict).train_test_split(test_size=0.2)


def prepare_model(args, intent_list):
    model = Qwen2ForSequenceClassification.from_pretrained(
        args.base_model_name_or_path,
        num_labels=len(intent_list),
    ).bfloat16()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
    )

    return get_peft_model(model, peft_config)


def tokenize_and_align_labels(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(
        examples["question"],
        padding="longest",
        max_length=max_length,
        truncation=True,
    )
    tokenized_inputs["labels"] = examples["intent"]
    return tokenized_inputs


def main():
    parser = HfArgumentParser(
        (ModelArguments, NLUTrainingArguments, DataArguments, InferArguments)
    )
    (
        model_args,
        training_args,
        data_args,
        infer_args,
    ) = parser.parse_args_into_dataclasses()

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_name_or_path,
        pad_token="<|endoftext|>",
        trust_remote_code=True,
    )
    if "qwen" in model_args.base_model_name_or_path.lower():
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    if data_args.dataset == "intent":
        intent_dict = {
            "港口停靠与航行规划": 0,
            "货物装卸与物流": 1,
            "船舶维护与检查": 2,
            "合规性与法律事务": 3,
            "船员管理与福利": 4,
            "与港口当局和其他利益相关者的沟通": 5,
            "船舶安全与应急准备": 6,
            "船舶运营与调度": 7,
            "市场与商业信息": 8,
            "船舶技术与系统": 9,
            "船舶环境与可持续性": 10,
            "船员培训与发展": 11,
            "船舶财务管理": 12,
            "船舶文化与社会责任": 13,
            "综合": 14,
        }
        ds = load_dataset_from_jsonl("./other/data.jsonl", intent_dict)
    else:
        raise NotImplementedError

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        accuracy = (preds == labels).mean()
        precision = precision_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds, average="weighted")
        f1 = f1_score(labels, preds, average="weighted")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    if training_args.do_train:
        model = prepare_model(model_args, list(intent_dict.keys()))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.print_trainable_parameters()

        tokenized_ds = ds.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, model_args.max_length),
            batched=True,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    if infer_args.do_infer:
        adapter_path = model_args.adapter_name_or_path
        peft_config = PeftConfig.from_pretrained(adapter_path)
        model = Qwen2ForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path, num_labels=len(intent_dict)
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        # merge and unload is necessary for inference
        model = model.merge_and_unload()
        model.config.pad_token_id = tokenizer.pad_token_id
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        def infer(question):
            inputs = tokenizer(
                question,
                padding="longest",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            return predictions.cpu().numpy()[0]

        question = "请问，在苏州迈为科技股份有限公司2019年的年报中，现金流的情况是否发生了重大变化？若发生，导致重大变化的原因是什么？"
        prediction = infer(question)
        intent_label = {v: k for k, v in intent_dict.items()}[prediction]
        print(f"Question: {question} -> Predicted Intent: {intent_label}")


if __name__ == "__main__":
    main()
