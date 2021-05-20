from transformers import BertModel, BertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
from transformers import LineByLineTextDataset
#from tokenizers import Tokenizer

configuration = BertConfig()
model = BertModel(configuration)
configuration = model.config

#tokenizer = Tokenizer.from_file("wiki_tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./wiki_tokenizer.json")
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./processed/processed_wiki_ko.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./KoBERT",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./KoBERT")