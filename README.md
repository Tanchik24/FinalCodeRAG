## Запуск проекта

### Требования
- Docker + Docker Compose

## Шаги

1) Склонируйте репозиторий:
```bash
git clone <REPO_URL>
cd <REPO_FOLDER>
```

2) Создайте и активируйте виртуальное окружение:
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows 
.venv\Scripts\Activate.ps1 
```

2) Установите зависимости:
```bash
pip install -r requirements.txt
```

3) Создайте .env на основе шаблона и добавьте ключ модели:
```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows
```
Откройте .env и вставьте ваш ключ 

4) Поднимите инфраструктуру через Docker:
```bash
docker compose up -d --build
```

5) Запустите приложение:
```bash
python main.py
```

6) Остановка
```bash
docker compose down
````

____

## Архитектура

Система строит два индекса по коду репозитория: **графовый** (Neo4j) и **векторный** (Qdrant). Поверх них работает **LangGraphтагент**, у которого есть **3 тулзы**:

* **`graph query`** - генерирует Cypher и ходит в **Neo4j** за структурной информацией: где что определено, связи, импорты, наследование и т.д.
* **`semantic search`** - запускает семантический поиск по эмеддингам в **Qdrant** и возвращает релевантные кусочки кода

* **`read_file_span`** - достает кусочки кода из файла по пути и start/end lines
Агент сам решает, какую тулзу вызвать (иногда обе), и может делать несколько итераций, пока контекста не хватит для ответа, используется create_react_agent из langgraph.prebuilt

---

## Компоненты

* **GraphBuilder** - парсит файлы/папки, извлекает сущности (модули/классы/функции/методы), докстринги, декораторы и связи
* **ImportsParser** - разбирает импорты и связывает модули отношениями `IMPORTS`
* **LanguageRegistry** - выбирает грамматику/парсер по расширению файла (Tree-sitter)
* **Neo4jIngestor** - батч-запись/чтение графа в Neo4j
* **CodeEmbeddingsGenerator** - извлекает код сущностей по `path + start/end_line` и считает эмбеддинги
* **CodeEmbeddingsStore** - хранит и ищет эмбеддинги в локальном Qdrant
* **CodebaseIndexer** - оркестрация индексации: граф -> flush -> эмбеддинги
* **CodeRepoToolAgent** - LangGraph-агент на Mistral, вызывает тулзы `graph_query`, `semantic_search`
и `read_file_span`
---

## Взаимодействие компонентов

**Индексация.** `CodebaseIndexer` запускает `GraphBuilder` (плюс `ImportsParser`) и записывает результат через `Neo4jIngestor` в Neo4j. Затем `CodeEmbeddingsGenerator` читает сущности из Neo4j, вытаскивает их код из файлов и сохраняет эмбеддинги в `CodeEmbeddingsStore` (Qdrant).

**Ответы.** `CodeRepoToolAgent` получает вопрос и выбирает стратегию: отправить запрос в Neo4j через `graph_query`, сделать семантический поиск через `semantic_search`, либо достает нужные куски кода через `read_file_span`

---

## Основные зависимости

* **Neo4j**: `neo4j`
* **Qdrant**: `qdrant-client`
* **LangGraph / LangChain**: `langgraph`, `langchain-core`, `langchain-mistralai`
* **Tree-sitter**: `tree-sitter`, `tree-sitter-languages`
* **Эмбеддинги**: `sentence-transformers`, `torch`, `numpy`
* **Утилиты**: `python-dotenv`, `tqdm`
