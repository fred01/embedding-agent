# Embedding Agent DevContainer

Контейнер для запуска агента на CPU-машинах флота: образ `ghcr.io/fred01/embedding-agent-devcontainer:latest`
(Python, CPU-only PyTorch, transformers, предзагруженная BGE-M3), код монтируется в workspace,
`CMD` в `Dockerfile` запускает `agent.py`, контейнер живёт, пока работает агент.

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `AGENT_TOKEN` | — | Токен индексера (`FACADE_TOKEN` в его секретах). Старое имя `RS_HTTP_FACADE_TOKEN` тоже читается |
| `INDEXER_URL` | `https://book-indexer.svc.fred.org.ru` | Откуда брать работу |
| `WORKER_NAME` | `<hostname>-cpu` | Уникальное имя агента: под ним индексер ведёт аренды и статистику |
| `DEVICE` | `cpu` | В этом контейнере всегда CPU |

Очереди (rs-http-facade, Redis) больше нет: агент сам забирает книги у индексера по HTTP.
Веб-дашборда у агента тоже больше нет: все агенты, их скорость и аренды видны на странице
«Векторизация» индексера.

## Пересборка базового образа

```bash
docker build -f .devcontainer/Dockerfile.base -t ghcr.io/fred01/embedding-agent-devcontainer:latest .
docker push ghcr.io/fred01/embedding-agent-devcontainer:latest
```

## Локально с Docker

```bash
docker run -it --rm \
  -e AGENT_TOKEN="..." -e WORKER_NAME="$(hostname)-cpu" \
  -v $(pwd):/workspace \
  ghcr.io/fred01/embedding-agent-devcontainer:latest \
  python3 -u agent.py
```

CPU считает BGE-M3 медленно (порядка 0.1 чанка/с на ядро). Для GPU — `Dockerfile.cuda` в корне репозитория,
для Mac — `./start.sh` (см. README).
