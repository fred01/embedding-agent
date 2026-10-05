# Embedding Agent

Агент считает эмбеддинги чанков книг для [search-lib](https://github.com/fred01/search-lib):
dense-векторы **BAAI/bge-m3** (1024, L2-нормированные), совместимые с уже лежащими в Qdrant.

## Как работает

Очереди нет. Агент сам забирает работу у индексера по HTTP:

1. `POST /api/agent/embeddings/lease` — индексер выдаёт книги (или начало большой книги) с текстами чанков,
   примерно на `TARGET_SECONDS` (по умолчанию 2 минуты) работы именно этого агента: медленный берёт мало,
   быстрый — много, заранее ничего не копится.
2. Агент считает векторы.
3. `POST /api/agent/embeddings/results` — индексер пишет векторы в Qdrant и переводит чанки в READY.
4. Следующая аренда запрашивается и результаты отправляются, пока модель занята, — GPU не ждёт сеть.

Если агент упал, его аренды истекают (время аренды — 3× ожидаемого времени работы по его же скорости,
минимум 10 минут), и книги выдаются заново. При остановке (Ctrl+C / SIGTERM) агент досчитывает текущую
пачку и сразу возвращает незаконченные книги. Повторно посчитанный чанк ничего не ломает: id точки в Qdrant
детерминированный.

Все агенты, их скорость, сколько у кого в работе и оценка остатка видны на странице «Векторизация» индексера.

## Совместимость векторов

Вектор считается через `transformers` ровно как `dense_vecs` у FlagEmbedding (CLS последнего слоя + L2-норма);
на CPU в fp32 совпадение с FlagEmbedding — косинус 1.000000. При старте агент считает эталонный текст из
`embeddings/reference.json` (вектор посчитан FlagEmbedding) и не запускается, если косинус ниже 0.995 —
так fp16 на GPU/Mac или другая версия библиотек не испортят базу незаметно. Индексер со своей стороны
принимает только `BAAI/bge-m3` размерности 1024.

## Запуск

```bash
AGENT_TOKEN=... ./start.sh
```

`start.sh` создаёт `venv`, ставит PyTorch под платформу и запускает `agent.py`:

| Платформа | Устройство | Точность | Батч |
|-----------|------------|----------|------|
| Linux + NVIDIA | `cuda` | fp16 | 32 |
| macOS, Apple Silicon (M1–M5) | `mlx` (GPU через MLX; `DEVICE=mps` — PyTorch) | fp16 | 16 |
| остальное | `cpu` | fp32 | 4 |

Замер скорости без подключения к индексеру: `./start.sh --benchmark 64`.

### Mac

```bash
git clone https://github.com/fred01/embedding-agent && cd embedding-agent
AGENT_TOKEN=... WORKER_NAME=macbook-m5max ./start.sh
```

Нужен Python 3.10+ (`brew install python`). Модель (~2.3 ГБ) скачивается в `~/.cache/huggingface` при первом
запуске. Mac не должен засыпать: `caffeinate -i ./start.sh`.

По умолчанию на Mac считает MLX: тот же проход XLM-RoBERTa, что в transformers, но с fused attention. На M5 Max
на свободном GPU — 41.8 чанка/с против 27.8 у PyTorch на MPS, косинус с эталоном 1.00000. При первом запуске веса
конвертируются в fp16 и кэшируются в `~/.cache/embedding-agent`. `DEVICE=mps` возвращает PyTorch; если какой-то
операции нет на MPS, она выполнится на CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1` ставится автоматически).

### NVIDIA в Docker

```bash
docker build -f Dockerfile.cuda -t embedding-agent:cuda .
docker run -d --gpus all --restart unless-stopped \
  -e AGENT_TOKEN=... -e WORKER_NAME="$(hostname)-gpu0" embedding-agent:cuda
```

Несколько GPU — по агенту на карту: `DEVICE=cuda:1`, свой `WORKER_NAME`.

### Внешний сервер эмбеддингов

Агент может не считать сам, а ходить в любой OpenAI-совместимый `/embeddings` с BGE-M3
(HuggingFace text-embeddings-inference, infinity, LiteLLM):

```bash
EMBED_URL=http://gpu-box:8080/v1 AGENT_TOKEN=... python agent.py
```

Эталонная проверка выполняется и в этом режиме.

### CPU-флот (devcontainer)

См. [.devcontainer/README.md](.devcontainer/README.md).

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `AGENT_TOKEN` | — | Токен индексера (`FACADE_TOKEN` в секретах деплоя). Читаются и старые `RS_HTTP_FACADE_TOKEN`, `FACADE_TOKEN` |
| `INDEXER_URL` | `https://book-indexer.svc.fred.org.ru` | Адрес индексера |
| `WORKER_NAME` | `<hostname>-<device>` | Уникальное имя агента: под ним индексер ведёт аренды и статистику |
| `DEVICE` | `auto` | `auto`, `mlx`, `cuda`, `cuda:N`, `mps`, `cpu` (`--cpu` и `FORCE_CPU=true` = `cpu`) |
| `FP16` | `auto` | fp16 на GPU/MPS, fp32 на CPU |
| `BATCH_SIZE` | по устройству | Размер батча модели; при нехватке памяти уменьшается сам |
| `TARGET_SECONDS` | `120` | Сколько секунд работы брать за одну аренду |
| `CPU_THREADS` | все ядра | Потоки PyTorch на CPU |
| `DUTY_CYCLE` | `1` | Доля времени, которую считает модель: `0.3` = 30% работы, 70% пауз между батчами. Для шумных машин: меньше нагрев — тише вентиляторы. Скорость аренды подстраивается сама |
| `DUTY_CYCLE_FILE` | — | Файл, число из которого перекрывает `DUTY_CYCLE` без перезапуска (перечитывается раз в 10 с): `echo 0.2 > /tmp/duty_cycle` |
| `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY` | — | Внешний сервер эмбеддингов вместо локальной модели |
| `SKIP_REFERENCE_CHECK` | `false` | Не сверять с эталоном при старте (не рекомендуется) |
