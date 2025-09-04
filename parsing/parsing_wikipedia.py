import wikipediaapi
import pandas as pd
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

wiki = wikipediaapi.Wikipedia(
    language='ru',
    user_agent='my_wiki_scraper/1.0 (veronika@example.com)'
)


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'\[\[.*?\|?(.*?)\]\]', r'\1', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()


def get_articles_from_category(category_name, max_articles=2000, min_paragraph_len=30):
    category = wiki.page("Категория:" + category_name)
    if not category.exists():
        print(f"Категория не найдена: {category_name}")
        return []
    articles = []
    count = 0

    def recurse(cat):
        nonlocal count
        members = list(cat.categorymembers.values())
        for c in members:
            if count >= max_articles:
                break
            if c.ns == wikipediaapi.Namespace.MAIN:
                text = clean_text(c.text)
                if len(text) > 50:
                    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) >= min_paragraph_len]
                    for p in paragraphs:
                        articles.append({'title': c.title, 'text': p})
                        count += 1
                        if count >= max_articles:
                            break
            elif c.ns == wikipediaapi.Namespace.CATEGORY:
                recurse(c)

    recurse(category)
    return articles


categories = [
    "литература", "аниме", "манга", "фанфики", "книги", "машинное_обучение",
    "поэзия", "проза", "драматургия", "философия", "мифология", "литературные_жанры",
    "художественная_литература", "литературоведение", "искусство", "культура",
    "живопись", "музыка", "театр", "кинематограф", "скульптура", "архитектура",
    "фотография", "романтизм", "символизм", "реализм", "сюрреализм", "импрессионизм",
    "экспрессионизм", "фантастика", "фэнтези", "детективы", "приключения", "любовные_романы"
]

all_articles = []

with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_category = {executor.submit(get_articles_from_category, cat, 5000): cat for cat in categories}

    for future in tqdm(as_completed(future_to_category), total=len(categories), desc="Категории"):
        cat = future_to_category[future]
        try:
            articles = future.result()
            all_articles.extend(articles)
        except Exception as e:
            print(f"Ошибка при сборе категории {cat}: {e}")

df = pd.DataFrame(all_articles)
df.to_csv("../../data/russian_fiction_dataset_parallel.csv", index=False)
print(f"Готово! Всего строк: {len(df)}")
