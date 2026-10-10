"""SGLang 0.5.21 for Gemma 4 31B + our LoRA, two fixes. Idempotent.
  1. The LoRA manager never calls the model's should_apply_lora gate (its own comment says VL models
     rely on it), so it tries to wrap vision-tower layers and fails on ClippableRowParallelLinear; and
     the memory pool's per-layer shape lookup takes the first module with that name and layer number,
     which is the vision tower's (8,608-wide MLP instead of the language model's 43,008).
  2. gemma4_mm.get_hidden_dim assumes one attention shape and a per-layer intermediate_size list (the
     small Gemma 4 models). The 31B's global layers (layer_types "full_attention") use head_dim 512
     and 4 KV heads, local layers 256 and 16, and intermediate_size is one int. LoRA buffers are
     allocated per layer, so a layer-aware get_hidden_dim is enough.
  python serving/sglang/patch_sglang.py     (in SGLang's environment)"""
import sglang.srt.lora.lora_manager as m

path = m.__file__
src = open(path).read()
anchor = '''            # The module should be converted if it is included in target_names
            parts = module_name.split(".")'''
gate = '''            # patch_sglang.py: honour the model's LoRA gate (VL models: language-model layers only)
            if hasattr(self.base_model, "should_apply_lora") and not self.base_model.should_apply_lora(module_name):
                continue
'''
if gate in src:
    print("already patched:", path)
else:
    assert src.count(anchor) == 1, "anchor not found: SGLang changed"
    open(path, "w").write(src.replace(anchor, gate + anchor))
    print("patched:", path)


import sglang.srt.models.gemma4_mm as g

path = g.__file__
src = open(path).read()
old_start = src.index("    def get_hidden_dim(self, module_name, layer_idx):")
old_end = src.index("    def get_embed(self):")
new = """    def get_hidden_dim(self, module_name, layer_idx):
        # patch_sglang.py: per-layer shapes read from the layer's own linear modules (Gemma 4 31B: local
        # layers 256-dim heads x 16 KV heads, global 512 x 4; SGLang's runtime config cannot tell them apart)
        layer = self.language_model.layers[layer_idx]
        lin = lambda m: getattr(m, "base_layer", m)
        mod = {"qkv_proj": lambda: layer.self_attn.qkv_proj, "o_proj": lambda: layer.self_attn.o_proj,
               "gate_up_proj": lambda: layer.mlp.gate_up_proj, "down_proj": lambda: layer.mlp.down_proj}.get(module_name)
        if mod is None:
            raise NotImplementedError()
        m = lin(mod())
        return m.input_size, m.output_size

"""
if "read from the layer's own linear modules" in src:
    print("already patched:", path)
else:
    open(path, "w").write(src[:old_start] + new + src[old_end:])  # replaces any earlier version of the patch
    print("patched:", path)


import sglang.srt.lora.mem_pool as mp

path = mp.__file__
src = open(path).read()
a = """                if layer_id is not None and (
                    hasattr(base_layer, "input_size_per_partition")"""
b = """                if layer_id is not None and (  # patch_sglang.py: the model's LoRA gate
                    not hasattr(base_model, "should_apply_lora") or base_model.should_apply_lora(name)
                ) and (
                    hasattr(base_layer, "input_size_per_partition")"""
if "patch_sglang.py: the model's LoRA gate" in src:
    print("already patched:", path)
else:
    assert src.count(a) == 1, "mem_pool anchor not found: SGLang changed"
    open(path, "w").write(src.replace(a, b))
    print("patched:", path)
