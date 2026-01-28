# Embedding Agent DevContainer

Этот devcontainer используется для запуска Embedding Agent в контейнеризованном окружении.

## Особенности конфигурации

### Keep-Alive команда

В `Dockerfile` используется команда:
```dockerfile
CMD ["python3", "-u", "agent_sse.py"]
```

Эта команда **держит контейнер живым**, так как `agent_sse.py` работает бесконечно в SSE loop, ожидая и обрабатывая сообщения. Контейнер будет работать до тех пор, пока:
- Агент не будет остановлен (Ctrl+C)
- Контейнер не будет остановлен вручную
- Произойдет критическая ошибка

### Переменные окружения

Все переменные можно переопределить через environment variables хост-системы:

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `FACADE_URL` | `http://localhost:8090` | URL rs-http-facade |
| `FACADE_TOKEN` | `default-token` | Токен аутентификации |
| `TASK_STREAM` | `embedding_tasks` | Название stream для задач |
| `TASK_GROUP` | `embedding-agent` | Группа consumer'а |
| `RESULT_STREAM` | `embedding_results` | Stream для результатов |
| `WEB_PORT` | `5000` | Порт веб-дашборда |
| `FORCE_CPU` | `true` | Принудительно использовать CPU |

### Порты

- **5000** - Веб-дашборд со статистикой (автоматически прокидывается наружу)

### Монтирование

- **Workspace** - `/workspace` (весь проект)
- **HuggingFace cache** - `~/.cache/huggingface` монтируется в контейнер для кеширования моделей

## Docker Image

DevContainer использует готовый образ из GitHub Container Registry:
```
ghcr.io/fred01/embedding-agent-devcontainer:latest
```

Образ содержит:
- Python 3.11
- Все зависимости из requirements.txt
- CPU-only PyTorch
- Предзагруженную модель BGE-M3 (~2GB)

### Пересборка образа

При изменении `requirements.txt` нужно пересобрать и запушить образ:

```bash
# Собрать образ
docker build -f .devcontainer/Dockerfile.base -t ghcr.io/fred01/embedding-agent-devcontainer:latest .

# Запушить в registry
docker push ghcr.io/fred01/embedding-agent-devcontainer:latest
```

## Использование

### С вашей системой

Передайте `devcontainer.json` вашей системе. Она должна:

1. Скачать образ из `ghcr.io/fred01/embedding-agent-devcontainer:latest`
2. Запустить контейнер с переменными окружения
3. CMD автоматически запустит `agent_sse.py`
4. Контейнер будет работать, пока работает агент

### Локально с VS Code

1. Установить переменные окружения:
```bash
export FACADE_TOKEN="your-real-token"
export FACADE_URL="https://your-facade-url"
```

2. Открыть проект в VS Code
3. Command Palette → "Dev Containers: Reopen in Container"
4. Веб-дашборд будет доступен на http://localhost:5000

### Локально с Docker

```bash
# Запустить с переменными окружения (образ скачается автоматически)
docker run -it \
  --name embedding-agent \
  --network host \
  -e FACADE_TOKEN="your-token" \
  -e FACADE_URL="https://nsq.fred.org.ru" \
  -v $(pwd):/workspace \
  ghcr.io/fred01/embedding-agent-devcontainer:latest \
  python3 -u agent_sse.py

# Веб-дашборд доступен на http://localhost:5000
```

### Переопределение переменных

При запуске через вашу систему, передайте переменные окружения:

```bash
FACADE_TOKEN="prod-token" \
FACADE_URL="https://prod-facade.example.com" \
TASK_STREAM="prod_embeddings" \
WEB_PORT="8080" \
your-system-command .devcontainer/devcontainer.json
```

## Архитектура

```
┌─────────────────────────────────────┐
│         Container                   │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   agent_sse.py (CMD)         │  │
│  │   - SSE Consumer Loop        │  │
│  │   - Embedding Computation    │  │
│  │   - Result Publishing        │  │
│  │   - Flask Web Server (5000)  │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   BGE-M3 Model               │  │
│  │   (~2GB, cached in volume)   │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
           ↕ SSE
┌─────────────────────────────────────┐
│      rs-http-facade                 │
│      (FACADE_URL)                   │
└─────────────────────────────────────┘
```

## Оптимизация

### CPU-only PyTorch

По умолчанию используется CPU-only версия PyTorch:
- Меньше размер образа (~2GB вместо ~5GB)
- Быстрее сборка
- Не требует NVIDIA драйверов

Для GPU версии измените в Dockerfile:
```dockerfile
# Вместо CPU-only:
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Кеширование моделей

HuggingFace модели кешируются в volume `~/.cache/huggingface`:
- При первом запуске модель скачивается (~2GB)
- При последующих запусках используется из кеша
- Экономия времени: 5-10 минут на загрузку

## Мониторинг

После запуска контейнера:

1. **Логи агента** - `docker logs -f embedding-agent`
2. **Веб-дашборд** - http://localhost:5000 (или WEB_PORT)
3. **Статистика API** - http://localhost:5000/api/stats

## Troubleshooting

### Контейнер сразу останавливается

Проверьте переменную `FACADE_TOKEN`:
- Должна быть установлена
- Должна быть валидной

Агент выйдет с ошибкой если токен не указан.

### Модель не загружается

- Убедитесь что есть интернет
- Проверьте место на диске (нужно ~5GB свободно)
- Проверьте что volume для HuggingFace cache монтируется корректно

### Web dashboard не доступен

- Проверьте что порт 5000 (или WEB_PORT) прокинут
- Проверьте `docker ps` - контейнер должен быть в статусе "Up"
- Проверьте логи: `docker logs embedding-agent`
