import time
import json
from typing import Optional, Dict
from urllib.parse import urljoin
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


class LitPrichalParser:
    """Парсер сайта ЛитПричал"""

    def __init__(self, base_url: str = "https://www.litprichal.ru"):
        self.base_url = base_url
        self.headers = {"User-Agent": UserAgent().random}
        self.timeout = 10

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Загружает страницу и возвращает BeautifulSoup-объект."""
        try:
            time.sleep(1)
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке {url}: {e}")
            return None

    def get_genres(self) -> Dict[str, Dict[str, str]]:
        """Парсит список жанров с главной страницы."""
        url = f"{self.base_url}/prose.php"
        soup = self._get_page(url)
        if not soup:
            return {}

        genres = {}
        genre_blocks = soup.find_all("div", class_="col-sm-6 col-md-4")

        print("Парсинг жанров:")
        for block in tqdm(genre_blocks, desc="Жанры"):
            for genre_link in block.find_all("a"):
                name = genre_link.text.strip()
                link = urljoin(self.base_url, genre_link.get("href"))
                genres[name] = {"link": link}

        return genres

    def _get_page_count(self, genre_url: str) -> int:
        """Определяет количество страниц в жанре."""
        soup = self._get_page(genre_url)
        if not soup:
            return 1

        pagination = soup.find("ul", class_="pagination")
        if not pagination:
            return 1

        pages = pagination.find_all("li")
        if not pages:
            return 1

        try:
            last_page = int(pages[-1].find("a").get("href")
                            .strip("/").split("/")[-1]
                            .replace("p", ""))
            return last_page
        except (ValueError, IndexError):
            return 1

    def get_books(self, genre_url: str) -> Dict[str, Dict[str, str]]:
        """Парсит книги из указанного жанра с учетом пагинации.

        Args:
            genre_url: URL страницы жанра
        """
        total_pages = self._get_page_count(genre_url)

        books_data = {}

        print(f"\nПарсинг книг в жанре {genre_url} (всего страниц: {total_pages}):")

        for page in range(1, total_pages + 1):
            page_url = f"{genre_url}/{f'p{str(page)}'}" if page > 1 else genre_url
            soup = self._get_page(page_url)
            if not soup:
                continue

            books = soup.find_all("div", class_="col-md-6 x2")

            for book in tqdm(books, desc=f"Страница {page}/{total_pages}"):
                try:
                    title = book.find("a", class_="bigList").text.strip()
                    link = self.base_url + book.find("a", class_="bigList").get("href")
                    author = book.find("a", class_="forum").text.strip()

                    if author not in books_data:
                        text = self.get_text(link)
                        books_data[author] = {
                            "link": link,
                            "title": title,
                            "text": text,
                            "genre_url": genre_url
                        }

                except Exception as e:
                    print(f"\nОшибка при парсинге книги: {e}")
                    continue

            time.sleep(1.5)

        return books_data

    def get_text(self, url: str) -> str:
        """Возвращает текст"""
        time.sleep(0.5)
        soup = self._get_page(url)
        if not soup:
            return ""

        text_blocks = soup.find_all("div", class_="col-md-12 x2")
        return self.clean_text(str(text_blocks[1]))

    def clean_text(self, html: str) -> str:
        """Очищает HTML от ненужных элементов и возвращает чистый текст"""
        soup = BeautifulSoup(html, 'html.parser')

        for element in soup(['iframe', 'img', 'script', 'style',
                             'div.video-blk', 'div.video-block',
                             'div.ads', 'div.advertisement']):
            element.decompose()

        for div in soup.find_all('div', class_=lambda x: x and 'hidden' in x):
            div.decompose()
        clean_text = soup.get_text(separator='\n', strip=True)
        lines = []
        for line in clean_text.split('\n'):
            line = line.strip()
            if line:
                lines.append(line)

        final_text = '\n'.join(lines)

        return final_text

    def save_to_json(self, data: Dict, filename: str = "data/парсинг/data_litprichal.json") -> None:
        """Сохраняет данные в JSON-файл."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'\nДанные сохранены в {filename}')

    def parse_all_in_genre(self) -> Dict[str, Dict]:
        """Парсит все жанры и книги в них."""
        genres = self.get_genres()
        result = {}
        print("\nПарсинг книг по всем жанрам:")
        for genre_name, genre_data in tqdm(genres.items(), desc="Общий прогресс"):
            books = self.get_books(genre_data["link"])
            result[genre_name] = books
            time.sleep(10)
        return genres


parser = LitPrichalParser()
books_data = parser.parse_all_in_genre()
parser.save_to_json(books_data)
