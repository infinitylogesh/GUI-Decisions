"""Make an SGLang-loadable copy of a slot adapter (Gemma 4 31B). The weights are unchanged except:

  target_modules  our regex (LoRA on the language model only) becomes a list of module names, which
                  SGLang requires; patch_sglang.py keeps it off the vision tower
  v_proj          Gemma 4's 10 global-attention layers have no v_proj: keys double as values
                  (attention_k_eq_v), and SGLang copies k_proj's base weights into the value slot. So
                  those layers get a v_proj LoRA equal to their k_proj LoRA: exact, not an approximation.

  python serving/sglang/convert_adapter.py runs/j_E3/adapter runs/j_E3/adapter_sgl
"""
import json, os, re, sys

from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
c = json.load(open(os.path.join(src, "adapter_config.json")))
c["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
json.dump(c, open(os.path.join(dst, "adapter_config.json"), "w"), indent=1)
w = load_file(os.path.join(src, "adapter_model.safetensors"))
add = {k.replace("k_proj", "v_proj"): v.clone() for k, v in w.items()
       if re.search(r"self_attn\.k_proj\.lora_[AB]\.weight$", k) and k.replace("k_proj", "v_proj") not in w}
w.update(add)
save_file(w, os.path.join(dst, "adapter_model.safetensors"))
print(f"wrote {dst}: target_modules as a list, {len(add) // 2} layers given a v_proj LoRA copied from k_proj")
