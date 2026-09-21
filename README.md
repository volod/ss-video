# ss-video

Video ingest and search for outdoor autonomy. Import package `selfsuvis`.

Pins [volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.2.0` (`ss-perception`,
`ss-mapping`, `ss-fusion`, `fusion-rt`) and [volod/ss-common](https://github.com/volod/ss-common)
tag `v0.2.1`.

| Sibling | Role |
| --- | --- |
| [volod/ss-fusion](https://github.com/volod/ss-fusion) | Research pipeline, perception/mapping packages, fusion-rt |
| [volod/ss-sens](https://github.com/volod/ss-sens) | IoT sensor mesh |
| [volod/ss-control](https://github.com/volod/ss-control) | Site ingress, SSO, and CA |
| [volod/ss-mlab](https://github.com/volod/ss-mlab) | nanochat and sslm |
| [volod/ss-common](https://github.com/volod/ss-common) | Contracts and `ss_kit` |

```bash
make up          # api + fusion-rt + worker + ui + qdrant
```

See [docs](docs/README.md), the [specification](docs/design/spec.md), and
[current implementation](docs/impl/current.md).
