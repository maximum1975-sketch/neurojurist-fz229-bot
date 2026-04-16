import os, json, re, random, time, sys, requests, asyncio, tempfile
from typing import List, Callable, Awaitable, Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader
from openai import OpenAI, APITimeoutError, APIConnectionError
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
import whisper

# ==================== БЛОК 2: КОНФИГУРАЦИЯ ====================
ANALIZATOR_ENABLE = False
MODEL = "qwen/qwen3-4b-2507"
EVALUATOR_MODEL = "qwen/qwen3-4b-2507"
CHUNK_k = 2
TEMPERATURE = 0.1
CREATE_NEW_DB = False
FAISS_DB_PATH = "faiss_db_229fz_md"
BASE_PATH = "229fz_clean.md"
BASE_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-bge-m3"
QUESTION_GEN_TIMEOUT = 20
ANALYZER_TIMEOUT = 30
ANSWER_GEN_TIMEOUT = 150
VALIDATOR_TIMEOUT = 60
MAX_VALIDATION_RETRIES = 3
MAX_TIMEOUT_RETRIES = 2
WHISPER_MODEL_SIZE = "turbo"
ERR_STR = "Нет соединения с LLM, свяжитесь с автором нейроюриста по ФЗ-229 Максимом по номеру +79257819422"
CLASSIFIER_TIMEOUT = 15  # таймаут классификатора вопроса

class LLMConnectionError(Exception):
    """Бросается при отсутствии связи с локальным LLM-сервером."""
    pass

# ==================== БЛОК 3: ЭМБЕДДИНГИ ====================
class LmStudioEmbeddings(Embeddings):
    def __init__(self, base_url: str, model: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request_with_retry(self, url: str, payload: dict) -> dict:
        for _ in range(MAX_TIMEOUT_RETRIES):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    ConnectionError, ConnectionRefusedError):
                raise LLMConnectionError(ERR_STR)
            except Exception:
                continue
        raise LLMConnectionError(ERR_STR)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [str(t) for t in texts]
        payload = {"input": texts, "model": self.model}
        data = self._request_with_retry(f"{self.base_url}/embeddings", payload)
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ==================== БЛОК 4: RAG-ПАЙПЛАЙН ====================
class RAGPipeline:
    def __init__(self, db_path=FAISS_DB_PATH):
        self.db_path = db_path
        self.db = None
        self.embeddings = LmStudioEmbeddings(base_url=BASE_URL, model=EMBEDDING_MODEL, timeout=30)
        self.client = OpenAI(base_url=BASE_URL, api_key="not-needed")

    def load_document(self, file_path=BASE_PATH):
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()[0].page_content

    def create_chunks(self, text):
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("# ", "Глава"), ("## ", "Статья")],
            strip_headers=False
        )
        raw_chunks = splitter.split_text(text)
        source_chunks = []
        for chunk in raw_chunks:
            if not chunk.page_content.strip():
                continue
            meta = chunk.metadata
            source_chunks.append(Document(
                page_content=chunk.page_content,
                metadata={
                    "chapter": meta.get("Глава", "unknown"),
                    "article": meta.get("Статья", "unknown"),
                    "source": BASE_PATH,
                    "chunk_len": len(chunk.page_content)
                }
            ))
        return source_chunks

    def create_vector_store(self, chunks):
        self.db = FAISS.from_documents(chunks, self.embeddings)
        return self.db

    def save_vector_store(self):
        if self.db:
            self.db.save_local(self.db_path)

    def load_vector_store(self):
        if os.path.exists(self.db_path):
            self.db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            return True
        return False

    def search_relevant_chunks(self, query, k=CHUNK_k):
        if not self.db:
            raise ValueError("Векторная база не инициализирована")
        return self.db.similarity_search(query, k=k)

# ==================== БЛОК 5: МНОГОШАГОВАЯ ЛОГИКА ====================
# Тип колбэка для уведомлений боту: async функция принимает строку
NotifyCallback = Optional[Callable[[str], Awaitable[None]]]

class NeurojuristLogic:
    def __init__(self, client, rag_pipeline):
        self.client = client
        self.rag = rag_pipeline
        self.history = []

    def _call_llm_with_retry(self, model, messages, temperature, timeout):
        for _ in range(MAX_TIMEOUT_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, timeout=timeout
                )
                return response.choices[0].message.content
            except APIConnectionError:
                raise LLMConnectionError(ERR_STR)
            except (APITimeoutError, Exception):
                continue
        raise LLMConnectionError(ERR_STR)

    def step1_analyze_query(self, query, chunks):
        chunk_content = "\n".join([doc.page_content for doc in chunks])
        prompt = f"""Ты — юридический аналитизатор. Проанализируй вопрос пользователя.
ВОПРОС: {query}
РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ ЗАКОНА (для контекста):
{chunk_content}
ВЕРНИ ТОЛЬКО JSON. НЕ ПРИДУМЫВАЙ СТАТЬИ.
{{
 "тип_ситуации": "возбуждение ИП / арест / обжалование / взыскание / сроки / иное",
 "ключевые_факты": ["факт1", "факт2"],
 "поисковые_темы": ["тема для поиска 1", "тема для поиска 2"],
 "стороны": ["взыскатель", "должник"],
 "требования": "суть требования"
}}"""
        return self._call_llm_with_retry(MODEL, [{"role": "user", "content": prompt}], TEMPERATURE, ANALYZER_TIMEOUT)

    def step2_generate_answer(self, query, analysis, chunks):
        chunk_content = "\n".join([doc.page_content for doc in chunks])
        prompt = f"""Ты — Нейроюрист по Федеральному закону №229-ФЗ «Об исполнительном производстве».
ЗАДАЧА: Сформулируй структурированный ответ с обязательным точным цитированием конкретных статей.
ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}
РЕЗУЛЬТАТЫ АНАЛИЗА (Шаг 1):
{analysis}
РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ ЗАКОНА:
{chunk_content}
ТРЕБОВАНИЯ К ОТВЕТУ:
Ссылайся на КОНКРЕТНЫЕ статьи и пункты ФЗ-229
Структурируй ответ (пункты, подпункты)
Точно цитируй названия статей, точно цитируй содержимое статей, не делай примечаний, не делай комментариев от себя — отвечай только на основе предоставленных фрагментов не пропуская слова
Если информации недостаточно — укажи это
ОТВЕТ:"""
        return self._call_llm_with_retry(MODEL, [{"role": "user", "content": prompt}], TEMPERATURE, ANSWER_GEN_TIMEOUT)

    def step3_validate(self, query, answer, chunks):
        chunk_content = "\n".join([doc.page_content for doc in chunks])
        prompt = f"""Ты — валидатор юридических ответов по ФЗ-229.
ЗАДАЧА: Проверь ответ на противоречия с текстом закона.
ВОПРОС: {query}
ОТВЕТ НЕЙРОЮРИСТА:
{answer}
ФРАГМЕНТЫ ЗАКОНА ДЛЯ ПРОВЕРКИ:
{chunk_content}
ОЦЕНИ:
Есть ли противоречия с текстом закона? (ДА/НЕТ)
Все ли цитированные статьи существуют в предоставленных фрагментах? (ДА/НЕТ)
Есть ли фактические ошибки? (ДА/НЕТ)
Рекомендация: (корректировать / ответ корректен)
ОТВЕТ В ФОРМАТЕ JSON:
{{
 "противоречия": "ДА/НЕТ",
 "статьи_существуют": "ДА/НЕТ",
 "фактические_ошибки": "ДА/НЕТ",
 "рекомендация": "корректировать/ответ корректен",
 "комментарий": "краткое пояснение"
}}"""
        return self._call_llm_with_retry(EVALUATOR_MODEL, [{"role": "user", "content": prompt}], 0.01, VALIDATOR_TIMEOUT)

    def extract_json(self, text):
        json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*?\})'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except Exception:
                continue
        return None

    def classify_query(self, query: str) -> dict:
        """
        Классифицирует входящее сообщение.
        Возвращает dict:
          {"тип": "по_теме"}                        — юридический вопрос по ФЗ-229
          {"тип": "не_по_теме", "ответ": "..."}     — бытовое/офтопик, готовый ответ
        """
        prompt = f"""Ты — помощник-диспетчер юридического бота по ФЗ-229 «Об исполнительном производстве».

Тебе пришло сообщение от пользователя. Определи, относится ли оно к теме исполнительного производства, судебных приставов, взыскания долгов, арестов имущества, алиментов и смежных правовых вопросов по ФЗ-229.

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ: {query}

Если сообщение — приветствие, прощание, благодарность, вопрос о возможностях бота или любой бытовой/офтопик вопрос — ответь на него коротко и дружелюбно от имени Нейроюриста по ФЗ-229.

ВЕРНИ ТОЛЬКО JSON без пояснений:
Если вопрос по теме ФЗ-229:
{{"тип": "по_теме"}}

Если вопрос НЕ по теме:
{{"тип": "не_по_теме", "ответ": "<твой короткий дружелюбный ответ пользователю>"}}"""
        raw = self._call_llm_with_retry(
            MODEL, [{"role": "user", "content": prompt}], 0.3, CLASSIFIER_TIMEOUT
        )
        result = self.extract_json(raw)
        if result and result.get("тип") in ("по_теме", "не_по_теме"):
            return result
        # Если модель не вернула валидный JSON — считаем вопрос по теме, чтобы не потерять
        return {"тип": "по_теме"}

    async def full_pipeline(
        self,
        query: str,
        bot_notify: NotifyCallback = None
    ) -> str:
        """
        Единый пайплайн: анализ → генерация → валидация с ретраями.
        bot_notify — async колбэк для отправки статусных сообщений пользователю.
        Тайминги и промежуточные ответы идут только в консоль.
        Возвращает финальный корректный ответ.
        """

        async def notify(text: str):
            print(text)
            if bot_notify:
                await bot_notify(text)

        # ── Шаг 0: поиск чанков ──────────────────────────────────────────
        chunks = await asyncio.to_thread(self.rag.search_relevant_chunks, query, CHUNK_k)

        # ── Шаг 1: анализ (один раз, до цикла генерации) ─────────────────
        await notify("❓ Вопрос пользователя: "+query)

        if ANALIZATOR_ENABLE:
            await notify("Анализ запроса...")
            t0 = time.time()
            analysis_raw = await asyncio.to_thread(self.step1_analyze_query, query, chunks)
            elapsed = time.time() - t0

            try:
                json_match = re.search(r'\{.*\}', analysis_raw, re.DOTALL)
                data_analysis = json.loads(json_match.group()) if json_match else {}

                # Консоль — подробно
                print(f"   Результаты анализа:")
                print(f"   Тип ситуации: {data_analysis.get('тип_ситуации', 'не определен')}")
                print(f"   Ключевые факты: {', '.join(data_analysis.get('ключевые_факты', []))}")
                print(f"   Поисковые темы: {', '.join(data_analysis.get('поисковые_темы', []))}")
                print(f"   Стороны: {', '.join(data_analysis.get('стороны', []))}")
                print(f"   Требования: {data_analysis.get('требования', 'не указаны')}")
                print(f"   Анализ занял {elapsed:.2f} сек")

                # Бот — красиво
                if bot_notify:
                    analysis_text = (
                        f"<b>Тип ситуации:</b> {data_analysis.get('тип_ситуации', 'не определен')}\n"
                        f"<b>Ключевые факты:</b> {', '.join(data_analysis.get('ключевые_факты', []))}\n"
                        f"<b>Поисковые темы:</b> {', '.join(data_analysis.get('поисковые_темы', []))}\n"
                        f"<b>Стороны:</b> {', '.join(data_analysis.get('стороны', []))}\n"
                        f"<b>Требования:</b> {data_analysis.get('требования', 'не указаны')}\n"
                        f"Анализ занял {elapsed:.2f} сек\n"
                    )
                    await bot_notify(analysis_text, parse_mode="HTML")

            except Exception:
                print(f"   Анализ (raw): {analysis_raw[:200]}...")
                if bot_notify:
                    await bot_notify(f"Анализ: {analysis_raw[:200]}...")

            analysis = analysis_raw
        else:
            analysis = ""

        # ── Шаги 2–3: генерация + валидация с ретраями ───────────────────
        final_answer = None
        attempt = 0

        while attempt < MAX_VALIDATION_RETRIES:
            attempt_label = f"попытка {attempt + 1}/{MAX_VALIDATION_RETRIES}"

            # Шаг 2: генерация
            print(f"Генерация ответа ({attempt_label})...")
            if bot_notify:
                if attempt == 0:
                    await bot_notify("Генерация ответа...")
                else:
                    await bot_notify("Уточняю ответ...")

            t1 = time.time()
            answer = await asyncio.to_thread(self.step2_generate_answer, query, analysis, chunks)
            elapsed2 = time.time() - t1
            print(f"   Ответ сгенерирован за {elapsed2:.2f} сек")
            await bot_notify("Ответ сгенерирован за "+f"{elapsed2:.2f}"+" сек")
            print(f"   Ответ (черновик):\n{answer}\n")

            # Шаг 3: валидация
            print(f"Валидация ответа ({attempt_label})...")
            if bot_notify:
                await bot_notify("Валидация ответа...")

            t2 = time.time()
            validation_raw = await asyncio.to_thread(self.step3_validate, query, answer, chunks)
            elapsed3 = time.time() - t2
            print(f"   Время проверки ответа {elapsed3:.2f} сек")
            await bot_notify("Время проверки ответа "+f"{elapsed3:.2f}"+" сек")
            validation = self.extract_json(validation_raw)
            rec = validation.get("рекомендация") if validation else None
            comment = validation.get("комментарий", "") if validation else ""

            if rec == "ответ корректен":
                print(f"   Валидация успешна. Ответ строго соответствует ФЗ-229.")
                final_answer = answer
                break
            else:
                print(f"   Валидация: требуется корректировка. Комментарий: {comment}")
                if attempt + 1 >= MAX_VALIDATION_RETRIES:
                    print(f"   Исчерпаны попытки ({MAX_VALIDATION_RETRIES}), отправляем последний вариант.")
                    await bot_notify("Ответ нейроюриста может содержать неточности") # вероятность выдачи этого сообщения стремится к нулю
                    final_answer = answer
                    break
                attempt += 1

        # ── Сохранение в историю ──────────────────────────────────────────
        self.history.append({"query": query, "answer": final_answer or "Ошибка генерации"})
        return final_answer or "Ошибка генерации"

# ==================== БЛОК 6: ИНТЕРФЕЙС ====================
class NeurojuristInterface:
    def __init__(self):
        self.rag = RAGPipeline(db_path=FAISS_DB_PATH)
        self.logic: Optional[NeurojuristLogic] = None
        self._whisper_model = None

    def initialize(self, create_new_db=False):
        if not create_new_db and self.rag.load_vector_store():
            print("Используется существующая векторная база")
        else:
            print("Создание новой векторной базы...")
            text = self.rag.load_document(BASE_PATH)
            chunks = self.rag.create_chunks(text)
            print(f"Создано чанков: {len(chunks)}")
            self.rag.create_vector_store(chunks)
            self.rag.save_vector_store()
        self.logic = NeurojuristLogic(self.rag.client, self.rag)
        print("Нейроюрист готов к работе")

    @property
    def whisper_model(self):
        """Ленивая загрузка Whisper — один раз при первом голосовом сообщении."""
        if self._whisper_model is None:
            print(f"Загрузка Whisper ({WHISPER_MODEL_SIZE})...")
            self._whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            print("Whisper загружен.")
        return self._whisper_model

    def transcribe(self, audio_path: str) -> str:
        return self.whisper_model.transcribe(audio_path, language="ru")["text"].strip()

    def generate_example_questions(self, count=5):
        try:
            all_docs = self.rag.db.similarity_search("статья закон норма", k=50)
            random.shuffle(all_docs)
            questions = []
            for doc in all_docs[:count]:
                content = doc.page_content[:200]
                prompt = (
                    f"На основе следующего фрагмента ФЗ-229 сформулируй короткий юридический вопрос "
                    f"(не более 15 слов). Фрагмент:\n{content}\n\nВопрос:"
                )
                response = self.logic._call_llm_with_retry(
                    MODEL, [{"role": "user", "content": prompt}], 0.5, QUESTION_GEN_TIMEOUT
                )
                questions.append(response.strip())
            return questions
        except Exception:
            return [
                "Какие сроки подачи исполнительного листа?",
                "Можно ли наложить арест на единственное жильё должника?",
                "Какой порядок взыскания алиментов?",
                "Что делать при бездействии судебного пристава?",
                "С каких доходов нельзя производить взыскание?"
            ]

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def convert_md_to_html(text: str) -> str:
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

def prepare_telegram_chunks(text: str) -> list[str]:
    text = text.replace("\n---\n", "\n").strip()
    if not text:
        return []

    lines = text.split('\n')
    blocks = []
    current_block = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('**') and '**' in stripped[2:]:
            if current_block:
                blocks.append('\n'.join(current_block).strip())
                current_block = []
        current_block.append(line)

    if current_block:
        blocks.append('\n'.join(current_block).strip())

    final_chunks = []
    for block in blocks:
        if len(block) <= 4096:
            if block:
                final_chunks.append(block)
        else:
            temp = block
            while len(temp) > 4096:
                cut = temp.rfind('.', 0, 4096)
                if cut == -1:
                    cut = temp.rfind(' ', 0, 4096)
                if cut == -1:
                    cut = 4096
                chunk = temp[:cut + 1].strip()
                if chunk:
                    final_chunks.append(chunk)
                temp = temp[cut + 1:].strip()
            if temp:
                final_chunks.append(temp)

    return final_chunks if final_chunks else [text]

def get_main_keyboard():
    kb = ReplyKeyboardBuilder()
    kb.button(text="📝 Сгенерировать вопросы")
    kb.button(text="📜 История диалога")
    kb.button(text="❓ Популярные вопросы")
    kb.button(text="🛑 Стоп")
    kb.adjust(2, 2)
    return kb.as_markup(resize_keyboard=True)

def get_questions_keyboard(questions: list):
    kb = InlineKeyboardBuilder()
    for i, q in enumerate(questions, 1):
        q_text = q if len(q) <= 100 else q[:100] + "..."
        kb.button(text=f"{i}. {q_text}", callback_data=f"question_{i-1}")
    kb.button(text="🔙 Назад в меню", callback_data="back_to_menu")
    kb.adjust(1)
    return kb.as_markup()

def get_back_keyboard():
    kb = InlineKeyboardBuilder()
    kb.button(text="🔙 Назад в меню", callback_data="back_to_menu")
    kb.adjust(1)
    return kb.as_markup()

POPULAR_QUESTIONS = [
    "Какие сроки подачи исполнительного листа?",
    "Можно ли наложить арест на единственное жильё?",
    "Как взыскать алименты?",
    "Что делать при бездействии пристава?",
    "С каких доходов нельзя взыскивать?"
]

MENU_BUTTONS = {"📝 Сгенерировать вопросы", "📜 История диалога", "❓ Популярные вопросы", "🛑 Стоп"}

# ==================== НАСТРОЙКА AIogram ====================
BOT_TOKEN = "MY_TOKEN"
storage = MemoryStorage()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=storage)
app = NeurojuristInterface()


# ── Единая функция отправки ответа пользователю ──────────────────────────────
async def send_answer(message: types.Message, question: str):
    """Классифицирует вопрос, затем запускает пайплайн или отвечает напрямую."""
    try:
        print(f"\n{'='*60}\nВопрос: {question}\n{'='*60}")

        # ── Классификация ─────────────────────────────────────────────
        classification = await asyncio.to_thread(app.logic.classify_query, question)
        print(f"   Классификация: {classification.get('тип')}")

        if classification.get("тип") == "не_по_теме":
            reply = classification.get("ответ", "Я специализируюсь только на вопросах по ФЗ-229 «Об исполнительном производстве». Задайте юридический вопрос!")
            print(f"   Ответ (не по теме): {reply}")
            await message.answer(reply, reply_markup=get_main_keyboard())
            return

        # ── Пайплайн по теме ──────────────────────────────────────────
        async def bot_notify(text: str, parse_mode: str = None):
            await message.answer(text, parse_mode=parse_mode)

        final_answer = await app.logic.full_pipeline(question, bot_notify=bot_notify)

        print("Финальный ответ нейроюриста:")
        answer_chunks = prepare_telegram_chunks(final_answer)
        for chunk in answer_chunks:
            try:
                await message.answer(convert_md_to_html(chunk), parse_mode="HTML")
                print(chunk)
            except Exception:
                await message.answer(chunk)

        await message.answer(
            "Напечатайте следующий вопрос, задайте его голосом или выберите команду:",
            reply_markup=get_main_keyboard()
        )

    except LLMConnectionError:
        print(ERR_STR)
        await message.answer(ERR_STR)
    except Exception as e:
        print(f"Ошибка в пайплайне: {e}")
        await message.answer(f"Ошибка: {e}")


# ── Startup ───────────────────────────────────────────────────────────────────
@dp.startup()
async def on_startup():
    await asyncio.to_thread(app.initialize, CREATE_NEW_DB)


# ── Команды ──────────────────────────────────────────────────────────────────
@dp.message(Command("start"))
@dp.message(F.text.startswith("/"))
async def cmd_start(message: Message):
    await message.answer(
        'Привет! Я Нейроюрист по ФЗ-229 "Об исполнительном производстве"\n\n'
        'Напечатайте свой вопрос, отправьте мне голосовое сообщение или выберите команду:',
        reply_markup=get_main_keyboard()
    )

@dp.message(Command("stop"))
async def cmd_stop(message: Message):
    await message.answer("Бот остановлен. Для перезапуска используйте /start")
    print("Бот остановлен пользователем")


# ── Кнопки меню ──────────────────────────────────────────────────────────────
@dp.message(F.text == "📝 Сгенерировать вопросы")
@dp.callback_query(F.data == "generate_questions")
async def cmd_generate(callback_or_message, state: FSMContext):
    if isinstance(callback_or_message, types.CallbackQuery):
        message = callback_or_message.message
        await callback_or_message.answer()
    else:
        message = callback_or_message

    await message.answer("🔄 Генерирую примеры вопросов...")
    try:
        questions = await asyncio.to_thread(app.generate_example_questions, 5)
        await state.update_data(questions=questions)
        await message.answer("📌 Выберите вопрос:", reply_markup=get_questions_keyboard(questions))
    except Exception as e:
        await message.answer(f"Ошибка генерации: {e}")


@dp.callback_query(F.data.startswith("question_"))
async def cb_question_selected(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    idx = int(callback.data.split("_")[1])
    data = await state.get_data()
    questions = data.get("questions", [])
    if idx < len(questions):
        await send_answer(callback.message, questions[idx])
    else:
        await callback.message.answer("Вопрос не найден.", reply_markup=get_back_keyboard())


@dp.message(F.text == "📜 История диалога")
@dp.callback_query(F.data == "show_history")
async def cmd_history(callback_or_message):
    if isinstance(callback_or_message, types.CallbackQuery):
        message = callback_or_message.message
        await callback_or_message.answer()
    else:
        message = callback_or_message

    if not app.logic.history:
        await message.answer("История диалога пуста.", reply_markup=get_back_keyboard())
        return

    hist = "<b>ИСТОРИЯ ДИАЛОГА</b>\n\n"
    for i, e in enumerate(app.logic.history, 1):
        hist += f"{i}. <b>Вопрос:</b> {e['query']}\n<b>Ответ:</b> {e['answer'][:200]}...\n\n"
    for chunk in prepare_telegram_chunks(hist):
        await message.answer(convert_md_to_html(chunk), parse_mode="HTML")
    await message.answer("Напечатайте следующий вопрос или выберите команду:", reply_markup=get_main_keyboard())


@dp.message(F.text == "❓ Популярные вопросы")
async def cmd_popular(message: Message):
    kb = InlineKeyboardBuilder()
    for i, q in enumerate(POPULAR_QUESTIONS, 1):
        kb.button(text=f"{i}. {q}", callback_data=f"popular_{i-1}")
    kb.button(text="🔙 Назад", callback_data="back_to_menu")
    kb.adjust(1)
    await message.answer("❓ Популярные вопросы:", reply_markup=kb.as_markup())


@dp.callback_query(F.data.startswith("popular_"))
async def cb_popular_selected(callback: types.CallbackQuery):
    await callback.answer()
    idx = int(callback.data.split("_")[1])
    if idx < len(POPULAR_QUESTIONS):
        await send_answer(callback.message, POPULAR_QUESTIONS[idx])


@dp.message(F.text == "🛑 Стоп")
async def cmd_stop_btn(message: Message):
    await message.answer("Работа остановлена. Выберите новую команду или задайте вопрос.",
                         reply_markup=get_main_keyboard())


@dp.callback_query(F.data == "back_to_menu")
async def cb_back(callback: types.CallbackQuery):
    await callback.answer()
    await callback.message.answer(
        'Привет! Я Нейроюрист по ФЗ-229 "Об исполнительном производстве"\n\n'
        'Напечатайте свой вопрос или выберите команду:',
        reply_markup=get_main_keyboard()
    )


# ── Голосовые сообщения ───────────────────────────────────────────────────────
@dp.message(F.voice)
async def handle_voice(message: Message):
    await message.answer("🎙️ Распознаю голосовое сообщение...")

    ogg_path = None
    wav_path = None
    try:
        # Скачиваем ogg от Telegram
        voice = message.voice
        file_info = await bot.get_file(voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_file:
            ogg_path = ogg_file.name
        await bot.download_file(file_info.file_path, destination=ogg_path)

        # Конвертируем в wav через ffmpeg (whisper лучше работает с wav)
        wav_path = ogg_path.replace(".ogg", ".wav")
        ret = os.system(f'ffmpeg -y -i "{ogg_path}" -ar 16000 -ac 1 "{wav_path}" -loglevel quiet')
        audio_path = wav_path if (ret == 0 and os.path.exists(wav_path)) else ogg_path

        # Транскрибируем
        print("Транскрибирование голосового сообщения...")
        question = await asyncio.to_thread(app.transcribe, audio_path)
        print(f"Распознан вопрос: {question}")

        if not question:
            await message.answer("Не удалось распознать речь. Попробуйте ещё раз или напишите вопрос текстом.")
            return

        # Показываем пользователю распознанный текст
        # await message.answer(f"<b>Вопрос:</b> {question}", parse_mode="HTML")

        # Обрабатываем по пайплайну
        await send_answer(message, question)

    except LLMConnectionError:
        print(ERR_STR)
        await message.answer(ERR_STR)
    except Exception as e:
        print(f"Ошибка обработки голоса: {e}")
        await message.answer(f"Ошибка распознавания голоса: {e}")
    finally:
        for path in [ogg_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


# ── Текстовые сообщения (вопросы пользователя) ───────────────────────────────
@dp.message()
async def handle_query(message: Message):
    if not message.text:
        return
    if message.text in MENU_BUTTONS or message.text.startswith("/"):
        return
    await send_answer(message, message.text)


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))