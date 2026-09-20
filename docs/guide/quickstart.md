# Quick Start

## Production (Docker)

```bash
make up          # api + fusion-rt + worker + ui + qdrant
```

API http://localhost:8000, fusion-rt http://localhost:8001, UI http://localhost:8501.

Full guide: [Quick Start — Production](quickstart-production.md).

## Local development

API, worker, and UI with hot-reload: [Quick Start — Local](quickstart-local.md).

## Next steps

- [Configuration](../reference/configuration.md)
- [MediaMTX streaming](../reference/streaming-mediamtx.md)
- [API reference](../reference/api.md)
- [Troubleshooting](../operations/troubleshooting.md)

Sensor mesh and research pipeline are separate repositories:
[volod/ss-sens](https://github.com/volod/ss-sens),
[volod/ss-fusion](https://github.com/volod/ss-fusion).
