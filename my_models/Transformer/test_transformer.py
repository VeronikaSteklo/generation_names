import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from my_models.Transformer.model.model import make_model, generate_title
from my_models.Transformer.data.tokenization import TikTokenizer

torch.serialization.add_safe_globals([TikTokenizer])

model, tokenizer = make_model(N=2, d_model=128, d_ff=256, h=4)


checkpoint = torch.load(
    "/Users/veronika_steklo/PycharmProjects/generation_names/models/rofl.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

if "tokenizer" in checkpoint:
    tokenizer = checkpoint["tokenizer"]

while True:
    print("Введите текст (для завершения введите пустую строку) или выход для завершения:")
    lines = []
    while True:
        line = input()

        if line.lower() in ['выход', 'exit', 'quit']:
            print("Завершение работы...")
            exit()

        if line == "":
            break
        lines.append(line)

    text = "\n".join(lines)
    text = text.lower()
    title = generate_title(model, tokenizer, text, device=device)
    print(f"Сгенерированное название: {title}\n\n")
