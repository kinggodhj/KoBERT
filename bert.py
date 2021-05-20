from transformers import BertModel, BertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data.dataset import Dataset
from tokenizers import Tokenizer
from tokenizers.processors import BertProcessing
from tokenizers.models import BPE
import pdb
import torch

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.examples = []
        f=open("./processed_wiki_ko.txt", 'r', encoding="utf-8")
        lines=f.readlines()
        for line in lines:
            if len(line)>0 and not line.isspace():
                t=line.split('. ')
                batch_encoding = tokenizer.encode_batch(t, add_special_tokens=True)
                self.examples += [x.ids for x in batch_encoding]

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i): #-> Dict[str, torch.tensor]:
        return torch.tensor(self.examples[i])

configuration = BertConfig()
model = BertModel(configuration)
configuration = model.config

#tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
#trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
#tokenizer.pre_tokenizer = Whitespace()
#files = ['./processed_wiki_ko.txt']
#tokenizer.train(files=files, trainer=trainer)

#tokenizer = Tokenizer.from_file("./wiki_tokenizer.json")
#fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="wiki_tokenizer.json")
tokenizer = Tokenizer.from_file("./wiki_tokenizer.json")
tokenizer.enable_truncation(max_length=512)

#tokenizer._tokenizer.post_processor = BertProcessing(
#        single="[CLS] $A [SEP]",
#        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
#        special_tokens=[
#             ("[CLS]", tokenizer.token_to_id("[CLS]")),
#             ("[SEP]", tokenizer.token_to_id("[SEP]")),
#        ],
#)

tokenizer.post_processor = BertProcessing(sep=("[SEP]", tokenizer.token_to_id("[SEP]")), cls=("[CLS]", tokenizer.token_to_id("[CLS]")))


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./processed_wiki.txt",
)

training_args = TrainingArguments(
    output_dir="./KoBERT",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./KoBERT")
