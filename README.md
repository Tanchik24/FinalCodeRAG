## Архитектура

Система строит два индекса по коду репозитория: **графовый** (Neo4j) и **векторный** (Qdrant). Поверх них работает **LangGraphтагент**, у которого есть **2 тулзы**:

* **`graph_query`** - генерирует Cypher и ходит в **Neo4j** за структурной информацией: где что определено, связи, импорты, наследование и т.д.
* **`semantic_search`** - запускает семантический поиск по эмеддингам в **Qdrant** и возвращает релевантные кусочки кода

Агент сам решает, какую тулзу вызвать (иногда обе), и может делать несколько итераций, пока контекста не хватит для ответа

---

## Компоненты

* **GraphBuilder** - парсит файлы/папки, извлекает сущности (модули/классы/функции/методы), докстринги, декораторы и связи
* **ImportsParser** - разбирает импорты и связывает модули отношениями `IMPORTS`
* **LanguageRegistry** - выбирает грамматику/парсер по расширению файла (Tree-sitter)
* **Neo4jIngestor** - батч-запись/чтение графа в Neo4j
* **CodeEmbeddingsGenerator** - извлекает код сущностей по `path + start/end_line` и считает эмбеддинги
* **CodeEmbeddingsStore** - хранит и ищет эмбеддинги в локальном Qdrant
* **CodebaseIndexer** - оркестрация индексации: граф -> flush -> эмбеддинги
* **CodeRepoToolAgent** - LangGraph-агент на Mistral, вызывает тулзы `graph_query` и `semantic_search`

---

## Взаимодействие компонентов

**Индексация.** `CodebaseIndexer` запускает `GraphBuilder` (плюс `ImportsParser`) и записывает результат через `Neo4jIngestor` в Neo4j. Затем `CodeEmbeddingsGenerator` читает сущности из Neo4j, вытаскивает их код из файлов и сохраняет эмбеддинги в `CodeEmbeddingsStore` (Qdrant).

**Ответы.** `CodeRepoToolAgent` получает вопрос и выбирает стратегию: отправить запрос в Neo4j через `graph_query`, сделать семантический поиск через `semantic_search`, либо комбинировать оба результата

---

## Основные зависимости

* **Neo4j**: `neo4j`
* **Qdrant**: `qdrant-client`
* **LangGraph / LangChain**: `langgraph`, `langchain-core`, `langchain-mistralai`
* **Tree-sitter**: `tree-sitter`, `tree-sitter-languages`
* **Эмбеддинги**: `sentence-transformers`, `torch`, `numpy`
* **Утилиты**: `python-dotenv`, `tqdm`
