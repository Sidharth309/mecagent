My choices:

I used BLIP-2 model for baseline since it is a pretrained vision to text transformer and can be quickly finetuned for cadquery language. I then added lora adapters on the 'q' and 'v' attention layers for efficient adaptation using limited compute. I used causal_lm since the generation task is text-based. Another enhancement I implemented was multi-view learning. The idea was to encode geometry better by forcing the model to learn from different perspectives — this is inspired by NeRF-style multi-view encoding.

Bottlenecks:

The model is very slow due to 3D rendering. I ran into a lot of tokenization errors because cadquery code is sensitive to indentation and formatting. I would have encoded the code structure more explicitly by adding a loss term to penalize invalid CQ primitives or sequences and explored token-level constraints using T5 grammar control if I had more time. BLIP-2 is not the most optimal for syntax sensitivity.

Possible enhancements if I had more time:

When implementing model architecture for Multi-View BLIP-2 I wanted to explore more options other than concatenating views horizontally. I had thought of finetuning the vision backbone (ViT) by processing each view individually, pool features (attention-weighted) and feed it into the same decoder. This would would have required editing the vision encoder forward pass with 'Perceiver' or 'ImageBind' style encodings
