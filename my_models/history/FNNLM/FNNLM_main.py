from my_models.history.FNNLM.model.utils import train_model, generate_title

model, dataset = train_model()

print("\n--- Тестирование генерации ---")
sample_text = "в данной работе мы исследуем методы машинного обучения для классификации"
predicted = generate_title(model, sample_text, dataset)

print(f"Входной текст: {sample_text}")
print(f"Сгенерированное название: {predicted}")