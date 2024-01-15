import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Definire il percorso del file di testo
file_path = "percorso_al_tuo_file.txt"  # Sostituire con il percorso del tuo file di testo

# Caricare il modello preaddestrato e il tokenizer
model_name = "allenai/llama2-7B-chat"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Caricare il testo dal file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Tokenizzazione del testo
tokenized_text = tokenizer.encode(text, return_tensors="pt")

# Creare un dataset per il fine-tuning
dataset = TextDataset(tokenized_text, tokenizer=tokenizer)

# Definire l'aggregatore dei dati per il language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Definire gli argomenti di addestramento
training_args = TrainingArguments(
    output_dir="./llama_fine_tuned",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Modificare il numero di epoche se necessario
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
)

# Inizializzare il trainer per il fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Avviare il fine-tuning
trainer.train()

# Salvare il modello fine-tuned
trainer.save_model("./llama7b-chat-q8_fine_tuned_model")