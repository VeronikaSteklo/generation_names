import asyncio
import logging

from aiogram import Dispatcher

from bot import bot_init

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    bot = bot_init.bot

    dp = Dispatcher()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())