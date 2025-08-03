import logging

from aiogram import Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery

from bot.statest.state import GenerateState

router = Router()
logger = logging.getLogger(__name__)

@router.message(Command("start"))
async def start(self, message: Message):
    await message.answer("привет")

@router.message(Command("generate"))
async def generate(self, message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Введите текст для генерации названия")
    await state.set_state(GenerateState.generate)

@router.callback_query(StateFilter(GenerateState.generate))
async def generate_callback(callback: CallbackQuery, state: FSMContext):
    text = callback.data
    await state.update_data(generate=text)
