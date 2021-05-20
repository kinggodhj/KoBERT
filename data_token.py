from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
#from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
#tokenizer.pre_tokenizer = Whitespace()
files = ['./processed/processed_wiki_ko.txt']
tokenizer.train(files, trainer)

tokenizer.save("wiki_tokenizer.json")