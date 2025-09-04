import tiktoken


class TikTokenizer:
    """Токенизатор, основанный на токенизаторе openai"""

    def __init__(self, model_name="gpt-4.1"):
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

    def encode(self, text, add_special_tokens=True):
        """Преобразование текста в токены-индексы"""
        ids = self.enc.encode(text)
        if add_special_tokens:
            ids = [self.token2idx["<s>"]] + ids + [self.token2idx["</s>"]]
        return ids

    def decode(self, ids):
        """Преобразование токенов по индексу в текст"""
        tokens = []
        for token_id in ids:
            if token_id in self._idx2token:
                token = self._idx2token[token_id]
                if token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.enc.decode([token_id]))
        return "".join(tokens)

    def token_2_idx(self, token):
        """Преобразование токена в индекс"""
        return self.token2idx.get(token, self.token2idx["<unk>"])

    def idx_2_token(self, token_id):
        """Получение токена по его индексу"""
        return self._idx2token.get(token_id, "<unk>")
