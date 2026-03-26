import torch
import my_models.history.RNN.config as config


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for text, titles in loader:
        text, titles = text.to(device), titles.to(device)

        optimizer.zero_grad()
        output = model(text, titles[:, :-1])

        output_dim = output.shape[-1]
        loss = criterion(output.reshape(-1, output_dim), titles[:, 1:].reshape(-1))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for text, titles in loader:
            text, titles = text.to(device), titles.to(device)
            output = model(text, titles[:, :-1])
            output_dim = output.shape[-1]
            loss = criterion(output.reshape(-1, output_dim), titles[:, 1:].reshape(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def generate_title(model, text, vocab, device, max_len=15):
    model.eval()
    with torch.no_grad():
        tokens = vocab.encode(text)
        if not tokens: return "..."

        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

        _, h = model.rnn(model.embedding(src_tensor))

        curr_idx = vocab.stoi[config.SOS_TOKEN]
        result = []

        for _ in range(max_len):
            input_tensor = torch.LongTensor([[curr_idx]]).to(device)
            output, h = model.rnn(model.embedding(input_tensor), h)

            next_word_idx = output.argmax(2).item()
            if next_word_idx == vocab.stoi[config.EOS_TOKEN]:
                break

            if next_word_idx < len(vocab.itos):
                result.append(vocab.itos[next_word_idx])

            curr_idx = next_word_idx

        return " ".join(result)