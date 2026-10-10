---
title: GUI-Decisions
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Computer-use steps read in one forward pass, no decoding
models:
  - google/gemma-4-31B-it
  - infinitylogesh/GUI-Decisions-31B-LoRA
  - infinitylogesh/GUI-Decisions-31B
---

Slot reading on Gemma 4 31B: a computer-use step (action + point) or a box, read from one forward pass
instead of decoded. See the [blog post](https://logeshumapathi.com/blog/2026/10/06/slot-reading.html).

## Example images

- `settings.png`, `flights.png`, `todo.png`: mock interfaces made for this demo.
- `wiki_bee.png`: a screenshot of the Wikipedia article [Honey bee](https://en.wikipedia.org/wiki/Honey_bee)
  (text CC BY-SA 4.0).
- `blog.png`: a screenshot of [logeshumapathi.com/blog](https://logeshumapathi.com/blog/).
- COCO val2017 photos from Flickr, CC BY 2.0:
  `coco_dog.jpg` ([source](http://farm1.staticflickr.com/92/260183330_494bd996d8_z.jpg)),
  `coco_desk.jpg` ([source](http://farm4.staticflickr.com/3523/3750770006_56fa24a8f7_z.jpg)),
  `coco_armchair.jpg` ([source](http://farm1.staticflickr.com/169/414738646_adafe60def_z.jpg)),
  `coco_cow.jpg` ([source](http://farm9.staticflickr.com/8303/7825237468_9345ce9a8e_z.jpg)),
  `coco_motorcycle.jpg` ([source](http://farm8.staticflickr.com/7407/9120226816_110da6cb9f_z.jpg)).
