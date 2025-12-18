# Embedding Agent

Агент для вычисления эмбеддингов книг в системе Library Search System.

## Описание

Агент подключается к серверу индексации, получает текстовые чанки из очереди, вычисляет эмбеддинги с помощью модели **BGE-M3** (BAAI/bge-m3) и отправляет результаты обратно на сервер для сохранения в pgvector.

### Основные возможности

- **Автоопределение ML-ускорителя**:
  - macOS M1/M2/M3: использует MPS (Metal Performance Shaders)
  - Linux с NVIDIA GPU: использует CUDA
  - Остальные платформы: CPU
- **Модель BGE-M3**: state-of-the-art embedding модель с 1024 измерениями
- **Минимальный вывод**: только важные сообщения (получение задачи, вычисление, отправка)
- **Автоматическая обработка ошибок**: задачи возвращаются в очередь при сбоях
- **Простая установка**: один скрипт для создания окружения и запуска

## Требования

- **Python 3.8+**
- **~5 GB свободного места** (для модели BGE-M3 и зависимостей)
- **ML-ускоритель** (опционально, но рекомендуется):
  - macOS: Apple Silicon (M1/M2/M3)
  - Linux: NVIDIA GPU с CUDA

## Установка и запуск

### Быстрый старт

```bash
# Клонировать репозиторий (если еще не сделано)
git clone https://github.com/your-org/search-lib.git
cd search-lib/embedding-agent

# Установить переменные окружения
export AGENT_TOKEN="your-secret-token"
export INDEXER_URL="http://localhost:8080"  # опционально, по умолчанию localhost:8080

# Запустить агент (скрипт сам создаст venv и установит зависимости)
./start.sh
```

При первом запуске скрипт:
1. Создаст виртуальное окружение Python
2. Определит вашу платформу (macOS/Linux, CPU/GPU)
3. Установит PyTorch с поддержкой доступных ускорителей
4. Установит остальные зависимости
5. Загрузит модель BGE-M3 (~2 GB)
6. Запустит агент

### Переменные окружения

- **`AGENT_TOKEN`** (обязательно) - токен аутентификации для доступа к API сервера
- **`INDEXER_URL`** (опционально) - URL сервера индексации (по умолчанию `http://localhost:8080`)
- **`POLL_INTERVAL`** (опционально) - интервал опроса очереди в секундах (по умолчанию `5`)
- **`BATCH_SIZE`** (опционально) - количество задач для обработки в одном батче (по умолчанию `10`)

### Ручная установка (если нужно)

```bash
# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установить PyTorch (выберите вариант для вашей платформы)

# macOS Apple Silicon (M1/M2/M3):
pip install torch torchvision torchaudio

# Linux с CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Установить остальные зависимости
pip install -r requirements.txt

# Запустить агент
export AGENT_TOKEN="your-secret-token"
python3 agent.py
```

## Использование

### Локальная разработка

```bash
# Запустить сервер индексации локально (в другом терминале)
cd ../
./gradlew :indexer:bootRun

# В этой директории запустить агент
export AGENT_TOKEN="default-token"  # должен совпадать с app.embedding.agent-token в application.yml
./start.sh
```

### Продакшен (Kubernetes)

```bash
# Получить токен из секретов Kubernetes
export AGENT_TOKEN=$(kubectl get secret book-search-secrets -n book-search -o jsonpath='{.data.agent-token}' | base64 -d)

# Указать URL продакшен сервера
export INDEXER_URL="https://book-indexer.svc.fred.org.ru"

# Запустить агент
./start.sh
```

### Запуск нескольких агентов

Вы можете запустить несколько агентов параллельно для ускорения обработки. Каждый агент будет обрабатывать свою порцию задач из очереди.

```bash
# Терминал 1
AGENT_TOKEN="token" ./start.sh

# Терминал 2
AGENT_TOKEN="token" ./start.sh

# Терминал 3
AGENT_TOKEN="token" ./start.sh
```

## Вывод в консоль

Агент выводит минимум информации:

```
[agent-a1b2c3d4] Device: cuda
[agent-a1b2c3d4] Loading BGE-M3 model...
[agent-a1b2c3d4] Model loaded successfully
[agent-a1b2c3d4] Starting agent, polling every 5s
[agent-a1b2c3d4] Task 12345 completed
[agent-a1b2c3d4] Task 12346 completed
[agent-a1b2c3d4] Task 12347 completed
```

Каждый агент получает уникальный ID при запуске для различения в логах.

## API взаимодействие

Агент использует следующие endpoints сервера индексации:

### POST /api/embeddings/lease
Получить задачи из очереди.

**Заголовки:**
- `X-Agent-Token: <token>`

**Тело запроса:**
```json
{
  "limit": 1,
  "ttlSeconds": 600,
  "agentId": "agent-abc123"
}
```

**Ответ (200 OK):**
```json
{
  "items": [
    {
      "id": 12345,
      "chunkUid": "base64-encoded-uid",
      "text": "Текст чанка для обработки...",
      "sqlitePath": "/data/shards/shard_123.sqlite",
      "localChunkId": 42,
      "rawSize": 2048,
      "leaseUntil": "2025-11-06T12:00:00Z"
    }
  ]
}
```

**Ответ (204 No Content):** Нет доступных задач.

### POST /api/embeddings/complete
Отправить вычисленный эмбеддинг.

**Заголовки:**
- `X-Agent-Token: <token>`

**Тело запроса:**
```json
{
  "id": 12345,
  "chunkUid": "base64-encoded-uid",
  "modelName": "BAAI/bge-m3",
  "dim": 1024,
  "dtype": "float32",
  "embedding": [0.123, -0.456, ...]
}
```

**Ответ (200 OK):**
```json
{
  "status": "completed"
}
```

### POST /api/embeddings/complete-batch
Отправить несколько вычисленных эмбеддингов за один запрос (оптимизировано для производительности).

**Заголовки:**
- `X-Agent-Token: <token>`

**Тело запроса:**
```json
{
  "items": [
    {
      "id": 12345,
      "chunkUid": "base64-encoded-uid",
      "modelName": "BAAI/bge-m3",
      "dim": 1024,
      "dtype": "float32",
      "embedding": [0.123, -0.456, ...]
    },
    {
      "id": 12346,
      "chunkUid": "base64-encoded-uid-2",
      "modelName": "BAAI/bge-m3",
      "dim": 1024,
      "dtype": "float32",
      "embedding": [0.789, -0.012, ...]
    }
  ]
}
```

**Ответ (200 OK):**
```json
{
  "successful": 2,
  "failed": 0,
  "errors": null
}
```

### POST /api/embeddings/fail
Вернуть задачу в очередь при ошибке.

**Заголовки:**
- `X-Agent-Token: <token>`

**Тело запроса:**
```json
{
  "id": 12345,
  "agentId": "agent-abc123",
  "error": "Out of memory"
}
```

**Ответ (200 OK):**
```json
{
  "status": "failed"
}
```

## Производительность

Скорость обработки зависит от:
- **Устройства**: GPU >> MPS > CPU
- **Длины текста**: чанки по ~600 слов обрабатываются за 0.5-5 секунд
- **Размера батча**: батчинг значительно ускоряет отправку данных на сервер (рекомендуется 10-20)

Примерные скорости:
- **NVIDIA RTX 3090**: ~1-2 сек/чанк (вычисление)
- **Apple M1 Pro**: ~3-5 сек/чанк (вычисление)
- **CPU (Intel i7)**: ~10-20 сек/чанк (вычисление)

### Оптимизация батчинга

Отправка данных в батчах значительно ускоряет обработку за счет амортизации накладных расходов на обновление HNSW индекса в PostgreSQL:
- **Один за раз** (`BATCH_SIZE=1`): 2-5 сек на отправку
- **Батч 10** (`BATCH_SIZE=10`): ~0.5-1 сек на отправку каждого
- **Батч 20** (`BATCH_SIZE=20`): ~0.3-0.5 сек на отправку каждого

Рекомендуемое значение: `BATCH_SIZE=10` для баланса между эффективностью и надежностью.

## Troubleshooting

### Ошибка "AGENT_TOKEN environment variable is required"
Установите переменную окружения:
```bash
export AGENT_TOKEN="your-token"
```

### Ошибка "torch not available" или проблемы с GPU
Убедитесь, что PyTorch установлен с поддержкой вашего ускорителя:
```bash
# Проверить версию PyTorch и доступные устройства
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

### Модель не загружается
Убедитесь, что есть ~5 GB свободного места и интернет-соединение для скачивания модели при первом запуске.

### Сервер недоступен
Проверьте, что сервер индексации запущен и доступен по указанному URL:
```bash
curl -H "X-Agent-Token: $AGENT_TOKEN" -X POST $INDEXER_URL/api/embeddings/lease -H "Content-Type: application/json" -d '{"limit":1}'
```

## Лицензия

См. LICENSE в корне репозитория.
