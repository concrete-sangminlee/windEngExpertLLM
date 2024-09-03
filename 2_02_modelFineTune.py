import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model

def load_jsonl_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def prepare_dataset(data):
    texts = []
    for item in data:
        message_text = " ".join([str(msg.get('content', '')) for msg in item['messages']])
        texts.append(message_text)
    return Dataset.from_dict({"text": texts})

def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if 'labels' not in inputs or inputs['labels'] is None:
            raise ValueError("Inputs should include 'labels' for computing loss.")

        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_fct = torch.nn.CrossEntropyLoss()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def fine_tune_model(train_dataset, model_name, output_dir):
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

    tokenized_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,  # Epoch 수 증가
        per_device_train_batch_size=8,  # 배치 크기 증가
        gradient_accumulation_steps=2,  # 대규모 배치처럼 학습
        learning_rate=3e-5,  # 학습률 조정
        save_steps=5_000,  # 체크포인트 저장 빈도 증가
        save_total_limit=3,  # 체크포인트 수 조정
        logging_dir='./logs',
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    data_file = "customData_augmented.jsonl"
    model_name = "MLP-KTLim/llama3-Bllossom"
    output_dir = "./fine_tuned_llama_B"

    data = load_jsonl_data(data_file)
    train_dataset = prepare_dataset(data)

    fine_tune_model(train_dataset, model_name, output_dir)