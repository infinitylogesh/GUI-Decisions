## How it works

The assistant turn is a fixed template; each slot ends in a placeholder:

```
action: ___ x: ___ y: ___ x2: ___ y2: ___ key: ___ sdir: ___ samt: ___
```

A slot's answer is the next-token distribution at the token before its placeholder, restricted to that
slot's labels: the 11 action words for `action`, and 256 bins for each coordinate, every bin a single
byte-like token. One forward pass over the whole template reads every slot at once. A LoRA trained with
cross-entropy against soft (Gaussian) targets puts the probability on the right bins, so the distribution
is meaningful: the heatmap is P(x) × P(y).

What to type, or a final answer, is free text: for those steps the base model (adapter switched off)
writes it.

Boxes work the same way: centre x, centre y, and a log-scaled width and height, four slots.

### Measured (blog post, Gemma 4 31B NVFP4 on vLLM, one RTX PRO 6000)

| how the step is decided | time per step |
| --- | --- |
| **slot read** | **146 ms** |
| vanilla, native `[y, x]` JSON (~20 tokens) | 472–546 ms |
| vanilla + thinking (130–230 tokens) | 2.6–3.1 s |

| this model (E3) | |
| --- | --- |
| ScreenSpot clicks (1,272) | 0.866 |
| computer-use steps, held-out AGUVIS (1,600) | 0.659 |
| RefCOCO val photo boxes, mean IoU (3,811) | 0.770 |
| COCO val2017 boxes, mean IoU (1,000) | 0.742 |

### Limits

Trained on web, desktop and mobile screenshots (AGUVIS, Wave-UI) and RefCOCO photos. It decides one step
at a time from the screenshot, the instruction and the previous actions you give it; it does not act on
anything. Expect mistakes on small targets and on interfaces unlike its training data.
