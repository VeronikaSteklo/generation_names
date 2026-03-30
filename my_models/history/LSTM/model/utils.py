import torch


def generate_title(model, vocab, device, text, max_gen_len=20):
    model.eval()
    tokens = vocab.encode(text)
    input_ids = [vocab.stoi["<SOS>"]] + tokens + [vocab.stoi["<SEP>"]]
    input_tensor = torch.tensor([input_ids]).to(device)

    generated = []

    with torch.no_grad():
        for _ in range(max_gen_len):
            logits, _ = model(input_tensor)

            last_token_logits = logits[0, -1, :]

            probs = torch.softmax(last_token_logits / 0.4, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == vocab.stoi["<EOS>"]:
                break

            generated.append(next_token)

            next_tensor = torch.tensor([[next_token]]).to(device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

    return vocab.decode(generated)