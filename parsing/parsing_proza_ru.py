import time
import json
import re
from typing import Optional, Dict
from datetime import datetime, timedelta
from urllib.parse import urljoin
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


class ProzaRuParser:
    """Парсер для сайта proza.ru"""

    def __init__(self, base_url: str = "https://proza.ru/texts/list.html", delay: float = 1.5):
        self.base_url = base_url
        self.headers = {"User-Agent": UserAgent().random}
        self.timeout = 10
        self.delay = delay
        self.output_file = "../data/temp_data/json/data_proza_ru.json"

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Загружает страницу и возвращает BeautifulSoup-объект."""
        try:
            print(f"Загружается: {url}")
            time.sleep(self.delay)
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке {url}: {e}")
            return None

    def get_all_forms(self) -> Dict[str, str]:
        """Получает названия и ссылки на разделы с малыми формами."""
        soup = self._get_page(self.base_url)
        if not soup:
            return {}

        works_block = soup.find('ul', attrs={'type': 'square', 'style': 'color:#404040'})
        all_forms = works_block.find_all('ul', attrs={'type': 'square'})
        data_all_forms = {}
        for form in all_forms:
            category = form.find_all('a')
            for link in category:
                title = link.text.strip()
                full_link = "https://www.proza.ru" + link['href']
                data_all_forms[title] = full_link

        return data_all_forms

    def get_text(self, url: str) -> str:
        soup = self._get_page(url)
        if not soup:
            return ""

        text = soup.find('div', attrs={'class': 'text'})
        if not text:
            return ""

        return self.clean_text(str(text))

    def clean_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['iframe', 'img', 'script', 'style',
                             'div.video-blk', 'div.video-block',
                             'div.ads', 'div.advertisement']):
            element.decompose()

        for div in soup.find_all('div', class_=lambda x: x and 'hidden' in x):
            div.decompose()

        clean_text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        return '\n'.join(lines)

    def get_works(self, url: str, category_title: str) -> Dict[str, Dict[str, str]]:
        """

        :param url:
        :param category_title:
        :return:
        """
        soup = self._get_page(url)
        if not soup:
            return {}

        works_block = soup.find_all('ul', attrs={'type': 'square', 'style': 'color:#404040'})
        data_small_works = {}

        for works_list in works_block:
            for work in works_list.find_all('li'):
                work_data = work.find('a')
                if not work_data:
                    continue
                try:
                    author = work.find('a', attrs={'class': 'poemlink'}).text.strip()
                    title = work_data.text.strip()
                    link = "https://www.proza.ru" + work_data['href']
                    text = self.get_text(link)

                    data_small_works[author] = {
                        'link': link,
                        'title': title,
                        'text': text
                    }
                    self._update_output_file(category_title, title, data_small_works[title])
                    time.sleep(self.delay)
                except Exception as e:
                    print(f"Ошибка при обработке произведения: {e}")
                    continue

        return data_small_works

    def parse_by_dates(self, start_date: str, end_date: str, category_title: str, topic: str) -> Dict[str, dict]:
        """
        Парсит материалы за указанный период в обратном порядке
        :param category_title:
        :param start_date: Дата начала в формате 'YYYY-MM-DD'
        :param end_date: Дата окончания в формате 'YYYY-MM-DD'
        :param topic: ID темы
        :return: Словарь с данными
        """
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        result = {}

        while current_date >= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"\nОбработка даты: {date_str}")

            day = current_date.strftime("%d")
            month = current_date.strftime("%m")
            year = current_date.strftime("%Y")

            url = f"{self.base_url}?day={day}&month={month}&year={year}&topic={topic}"
            data_day = self.get_works(url, category_title)
            result.update(data_day)

            current_date -= timedelta(days=1)

        return result

    def _update_output_file(self, category: str, title: str, work_data: dict):
        """Обновляет JSON-файл, добавляя новое произведение"""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            if category not in existing_data:
                existing_data[category] = {}
            existing_data[category][title] = work_data

            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"❌ Ошибка при обновлении файла: {e}")

    def get_all_work(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Получает все малые формы и произведения внутри них."""
        all_data = {}
        all_forms = self.get_all_forms()

        for all_form_title, all_form_link in all_forms.items():
            print(f"\nОбработка категории: {all_form_title}")
            works = self.get_works(all_form_link, all_form_title)
            topic = re.search(r'topic=(\d+)', all_form_link).group(1)
            works_by_data = self.parse_by_dates("2025-07-18", "2025-07-11", all_form_title, topic)
            merged_works = {**works, **works_by_data}
            all_data[all_form_title] = merged_works
            time.sleep(self.delay)

        return all_data

    def save_to_json(self, data: dict, filename: str = "data/data_proza_ru.json"):
        """Сохраняет данные в JSON-файл."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"\n✅ Данные сохранены в {filename}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении JSON: {e}")


parser = ProzaRuParser()
parser.get_all_work()
