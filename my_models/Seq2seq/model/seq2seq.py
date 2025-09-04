import random
import time

import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_size == decoder.hid_dim, "Hidden dimensions must match!"
        assert decoder.n_layers == 1, "Encoder must produce compatible layers for decoder"

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.1):
        batch_size, trg_len = src.shape[0], trg.shape[1]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        _, (hidden, cell) = self.encoder(src)
        input_tok = trg[:, 0]

        for t in range(1, trg_len):
            out_step, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t] = out_step
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = out_step.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1
        return outputs

    # ---- train / eval ----
    def train_epoch(self, dataloader, optimizer, criterion, clip: float = 1.0):
        self.train()
        epoch_loss = 0.0
        for src, trg in dataloader:
            src, trg = src.to(self.device), trg.to(self.device)
            optimizer.zero_grad()
            outputs = self(src, trg)
            out_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, out_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(outputs, trg_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for src, trg in dataloader:
                src, trg = src.to(self.device), trg.to(self.device)
                output = self(src, trg, teacher_forcing_ratio=0.0)
                out_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, out_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg_flat)
                epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def fit(self, train_loader, val_loader, optimizer, criterion, scheduler,
            num_epochs=10, clip=1.0, early_stopping_patience=3, model_save_path='models/best_model_seq2seq.pt'):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        prev_val = None
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, optimizer, criterion, clip)
            val_loss = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            if prev_val is not None and (abs(val_loss - prev_val) <= 0.01 or val_loss > prev_val):
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
            prev_val = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), model_save_path)

            print(
                f"Epoch {epoch:02} | Train {train_loss:.3f} | Val {val_loss:.3f} | LR {optimizer.param_groups[0]['lr']:.6f} | Time {time.time() - t0:.1f}s")
            if epochs_no_improve >= early_stopping_patience:
                print(f"Ранний останов после {epoch:02} эпох.")
                break
        self.load_state_dict(torch.load(model_save_path))
        return best_val_loss

    # ---- generation / metrics ----
    def generate_sequence(self, src_sequence, src_vocab, trg_vocab, max_len: int = 20):
        import nltk
        self.eval()
        if isinstance(src_sequence, str):
            tokens = nltk.word_tokenize(src_sequence.lower())
        else:
            tokens = src_sequence
        src_indexes = [src_vocab.word2index.get(tok, src_vocab.word2index['<unk>']) for tok in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, (hidden, cell) = self.encoder(src_tensor)
        trg_indexes = [trg_vocab.word2index['<sos>']]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
            pred = output.argmax(1).item()
            trg_indexes.append(pred)
            if pred == trg_vocab.word2index.get('<eos>', -1):
                break
        tokens_out = []
        for idx in trg_indexes[1:]:
            tok = trg_vocab.index2word.get(idx, '<unk>')
            if tok != 'eos':
                tokens_out.append(tok)
        return tokens_out

    def calculate_bleu(self, dataloader, src_vocab, trg_vocab, max_len: int = 20):
        self.eval()
        refs, hyps = [], []
        smoothing = SmoothingFunction().method4
        with torch.no_grad():
            for src, trg in dataloader:
                src, trg = src.to(self.device), trg.to(self.device)
                output = self(src, trg, teacher_forcing_ratio=0.0).argmax(dim=-1)
                for i in range(trg.size(0)):
                    ref_idxs = trg[i].cpu().numpy()
                    ref_tokens = [trg_vocab.index2word.get(int(idx), '<unk>') for idx in ref_idxs if
                                  trg_vocab.index2word.get(int(idx), '<unk>') not in ['sos', 'eos', '<pad>']]
                    hyp_idxs = output[i].cpu().numpy()
                    hyp_tokens = []
                    for idx in hyp_idxs:
                        tok = trg_vocab.index2word.get(int(idx), '<unk>')
                        if tok == 'eos':
                            break
                        if tok not in ['sos', '<pad>']:
                            hyp_tokens.append(tok)
                    refs.append([ref_tokens])
                    hyps.append(hyp_tokens)
        return corpus_bleu(refs, hyps, smoothing_function=smoothing)
