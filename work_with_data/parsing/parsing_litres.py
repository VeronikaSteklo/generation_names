import time
import json
from typing import Optional, Dict
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

link = "https://www.litres.ru/genre/klassicheskaya-literatura-5028/"


class LitresParser:
    """Парсер сайта ЛитРес"""

    def __init__(self, url: str):
        self.litres_url = "https://www.litres.ru"
        self.base_url = url
        self.headers = {"User-Agent": UserAgent().random}
        self.timeout = 10
        self.filename = "/data/temp_data/litres.json"

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Загружает страницу и возвращает BeautifulSoup-объект."""
        try:
            time.sleep(1.5)
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке {url}: {e}")
            return None

    def get_books(self, url: str, pages: int = 800):
        """Парсит книги с учетом пагинации и сохраняет после каждой страницы."""
        books_data = self._load_existing_data()

        for page in tqdm(range(1, pages + 1)):
            page_url = f"{url}?page={page}"
            soup = self._get_page(page_url)
            if not soup:
                continue

            books = soup.find_all(
                "div", class_="Art-module__3wrtfG__content Art-module__3wrtfG__content_full"
            )
            time.sleep(1)

            for book in books:
                try:
                    info = book.find("a", class_="ArtInfo-module__Y-DtKG__title")
                    title = info.text.strip()
                    link = self.litres_url + info.get("href")

                    if title in books_data:
                        continue

                    text = self.get_text(link)
                    books_data[title] = text

                    time.sleep(0.5)

                except Exception as e:
                    print(f"\nОшибка при парсинге книги: {e}")
                    continue

            self.save_to_json(books_data)
            time.sleep(2)

        return books_data

    def get_text(self, url: str):
        """Получает текст - аннотацию"""
        soup = self._get_page(url)
        if not soup:
            return None

        block = soup.find("div", class_="BookDetailsAbout-module__p8ABVW__truncate")
        if not block:
            return None

        truncated = block.find("div", class_="Truncate-module__FwxwPG__truncated")
        return truncated.text if truncated else None

    def save_to_json(self, data: Dict, filename: str = None) -> None:
        """Сохраняет данные в JSON-файл."""
        filename = filename or self.filename
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _load_existing_data(self) -> Dict:
        """Загружает уже сохраненные данные, чтобы дописывать новые."""
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


parser = LitresParser(link)
books = parser.get_books(link)