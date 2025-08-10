import tiktoken


class TikTokenizer:
    """
        Токенизатор, основанный на токенизаторе openai
    """
    def __init__(self, model_name="gpt-4"):
        self.enc = tiktoken.encoding_for_model(model_name)

        self.token2idx = {"<unk>": 0}

        for i in range(1, self.enc.n_vocab):
            try:
                tok_str = self.enc.decode_single_token_bytes(i).decode("utf-8", errors="ignore")
            except Exception:
                tok_str = f"<unk_{i}>"
            self.token2idx[tok_str] = i

        self.special_tokens = {
            "<s>": self.enc.n_vocab,
            "</s>": self.enc.n_vocab + 1,
            "<pad>": self.enc.n_vocab + 2,
            "<mask>": self.enc.n_vocab + 3
        }
        self.token2idx.update(self.special_tokens)

        self._idx2token = {v: k for k, v in self.token2idx.items()}

        self.vocab_size = max(self._idx2token.keys()) + 1

    def encode(self, text):
        """Преобразование текста в числовые токены"""
        return self.enc.encode(text)

    def decode(self, ids):
        """Преобразование числовых токенов в текст"""
        tokens = []
        for token_id in ids:
            if token_id in self._idx2token:
                token = self._idx2token[token_id]
                tokens.append(token)
            else:
                tokens.append(self.enc.decode([token_id]))
        return "".join(tokens)

    def convert_token_to_id(self, token):
        """Преобразование токена в число"""
        return self.token2idx.get(token, self.token2idx["<unk>"])

    def convert_id_to_token(self, token_id):
        """Получение токена по его индексу"""
        return self._idx2token.get(token_id, self._idx2token[0])


tokenizer = TikTokenizer()
print("Vocab size:", tokenizer.vocab_size)
print("Decode <unk>:", tokenizer.decode([0]))
print("Decode 12561:", tokenizer.decode([12561]))
print("Decode 28089:", tokenizer.decode([28089]))
print("Decode 8341:", tokenizer.decode([8341]))
print("Encode Russian:", tokenizer.encode("Hello, привет, как дела"))
print("Decode Russian:", tokenizer.decode(tokenizer.encode("Hello, привет, как дела djhf авол")))

