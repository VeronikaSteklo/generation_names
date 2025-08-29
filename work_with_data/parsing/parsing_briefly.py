import random
import time
import json
from typing import Optional, Dict

from tqdm import tqdm

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

link = "https://briefly.ru/cultures/"


class ParsingBriefly():
    """Парсинг сайта Брифли с краткими пересказами произведений"""

    def __init__(self, url: str):
        self.briefly_url = "https://briefly.ru"
        self.base_url = url
        self.headers = {"User-Agent": UserAgent().random}
        self.timeout = 5
        self.filename = "../../data/temp_data/json/briefly.json"

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Загружает страницу и возвращает BeautifulSoup-объект."""
        try:
            self.human_delay()
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке {url}: {e}")
            return None

    def get_authors(self, url: str, name_culture: str = None):
        """Загружает всех авторов и их произведения"""
        soup = self._get_page(url)
        all_data = {}
        if soup is None:
            return None

        alphabet_index = soup.find("div", class_="index alphabet")
        if alphabet_index is None:
            return None

        authors = alphabet_index.find_all("a", class_="author")
        for author in tqdm(authors, postfix=name_culture):
            link = self.briefly_url + author.get("href")
            author_data = self.get_works(link)
            if author_data:  # защита
                all_data.update(author_data)

        return all_data

    def get_works(self, url: str):
        """Загружает произведения автора"""
        soup = self._get_page(url)
        if soup is None:
            return {}

        works_data = {}
        try:
            section = soup.find("section", class_="works_index")

            if section is None:
                section = soup.find("section", class_="author_works")
                if section is None:
                    return works_data

                works = section.find_all("div", class_="w-featured")
                for work in works:
                    title_el = work.find("div", class_="w-title")
                    link_el = work.find("a")
                    if not title_el or not link_el:
                        continue

                    title = title_el.text.strip()
                    link = self.briefly_url + link_el.get("href")
                    if any(word in title.lower() for word in ["глава", "том", "действие"]):
                        continue

                    text = self.get_text(link, category="published")
                    works_data[title] = text
                    self.human_delay(base=1.5, var=1.0)
            else:
                full_retelling = section.find_all("li", class_="published")
                small_retelling = section.find_all("li", class_="pending")

                for work in small_retelling:
                    work = work.find("a", class_="title")
                    if not work:
                        continue
                    title = work.text.strip()
                    link = work.get("href")
                    if any(word in title.lower() for word in ["глава", "том", "действие"]):
                        continue
                    text = self.get_text(link, category="pending")
                    works_data[title] = text
                    self.human_delay(base=1.5, var=1.0)

                for work in full_retelling:
                    work = work.find("a", class_="title")
                    if not work:
                        continue
                    title = work.text.strip()
                    link = self.briefly_url + work.get("href")
                    if any(word in title.lower() for word in ["глава", "том", "действие"]):
                        continue
                    text = self.get_text(link, category="published")
                    works_data[title] = text
                    self.human_delay(base=1.5, var=1.0)

            return works_data
        except Exception as e:
            print(f"Ошибка при парсинге {url}: {e}")
            return {}

    def get_text(self, url: str, category: str):
        """Получает текст произведения"""
        soup = self._get_page(url)
        if soup is None:
            return "Описание отсутствует"

        try:
            if category == "pending":
                element = soup.find("div", class_="microsummary__content")
            else:
                element = soup.find("p", class_="microsummary__content")

            if element:
                text = element.get_text(strip=True)
            else:
                main_div = soup.find("div", id="text")
                if main_div:
                    for ad in main_div.find_all("div", class_="honey"):
                        ad.decompose()
                    paragraphs = [p.get_text(" ", strip=True) for p in main_div.find_all("p")]
                    text = " ".join(paragraphs)
                else:
                    text = ""
            return text if text else "Описание отсутствует"
        except Exception as e:
            print(f"Ошибка в get_text {url}: {e}")
            return "Описание отсутствует"

    def get_all_data(self, url: str):
        all_data = self._load_existing_data()
        soup = self._get_page(url)
        try:
            cultures_cards = soup.find_all("a", class_="visited-hidden")
            for culture in cultures_cards[7:]:
                name = culture.get_text(strip=True)
                link = self.briefly_url + culture.get("href")
                data_culture = self.get_authors(link, name)
                if data_culture:
                    all_data.update(data_culture)

                    self.save_to_json(all_data)
                    self.human_delay(base=2, var=1.5)
        except Exception as e:
            print(e)

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

    @staticmethod
    def human_delay(base: float = 1.5, var: float = 1.0, long_pause_prob: float = 0.05):
        """Делает более человеческие задержки"""
        delay = random.uniform(base, base + var)
        time.sleep(delay)

        if random.random() < long_pause_prob:
            long_delay = random.uniform(5, 15)
            time.sleep(long_delay)


parser = ParsingBriefly(link)
parser.get_all_data(link)
