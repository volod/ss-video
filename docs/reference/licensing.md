# Licensing Notes

- OpenCLIP: check the model + weight license you choose.
- DINOv2: typically OK for research and commercial use, verify the specific model license.
- DINOv3: license ambiguity; use only if you accept the risk.

Planned 4D analysis candidates (Grounding DINO, CountGD, SAM 2/3, Depth Anything,
Metric3D, Perspective Fields, compact scene-graph VLMs, and local Qwen-VL-family models) are not
automatically approved by citation or by an available package. Before a model is pinned, record the
code license, weight license, dataset/use restrictions, redistribution terms, model revision and
digest, and whether inference sends frames outside the deployment boundary. If a compliant model
does not meet the declared quality, memory, and latency gates, a documented no-go is the expected
result.

This POC does not ship model weights; you download them separately.

---
[← Troubleshooting](../operations/troubleshooting.md) | [Tests →](../guide/tests.md)
