# Study Resources

Foundations behind the indexer, retrieval, and maps. Read these after a first
successful index job, not before.

## Layer 1 -- how machines see

- Goodfellow, Bengio, Courville, *Deep Learning* (2016), ch. 9 and 15.
  [deeplearningbook.org](https://www.deeplearningbook.org)
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020).
  [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- Radford et al., CLIP (2021). [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- He et al., MoCo (2020). [arxiv.org/abs/1911.05722](https://arxiv.org/abs/1911.05722)
- Vaswani et al., "Attention Is All You Need" (2017).
  [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

HuggingFace API: [transformers](https://huggingface.co/docs/transformers),
[models](https://huggingface.co/models), [optimum](https://huggingface.co/docs/optimum)
(ONNX export used by fine-tune jobs).

## Layer 2 -- geometry and maps

- Szeliski, *Computer Vision* (2nd ed.). [szeliski.org/Book](https://szeliski.org/Book)
- Cadena et al., SLAM survey (2016). [arxiv.org/abs/1606.05830](https://arxiv.org/abs/1606.05830)
- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), ch. 2-4 -- Kalman and
  uncertainty, used by fusion-rt and ss-fusion pose filters.

## Layer 3 -- labeling and adaptation

- Settles, "Active Learning Literature Survey" (2009) -- conceptual basis for
  CVAT / `al_tag` loops.
- Ren et al., "A Survey of Deep Active Learning" (2021).
  [arxiv.org/abs/2009.00236](https://arxiv.org/abs/2009.00236)

Research-pipeline math (Kalman/RTS, SSL, distillation) is in ss-fusion chapters
09-15 ([siblings.md](siblings.md)).

Next: [future directions](07_future_directions.md).
