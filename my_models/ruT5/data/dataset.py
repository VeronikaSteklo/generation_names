from torch.utils.data import Dataset


class T5TitleDataset(Dataset):
    def __init__(self, df, tokenizer, max_src_len=512, max_tgt_len=32):
        self.df = df
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        input_text = "make_title: " + str(row['text'])
        target_text = str(row['title'])

        source = self.tokenizer(
            input_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        target = self.tokenizer(
            target_text,
            max_length=self.max_tgt_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": source["input_ids"].flatten(),
            "attention_mask": source["attention_mask"].flatten(),
            "labels": target["input_ids"].flatten()
        }
