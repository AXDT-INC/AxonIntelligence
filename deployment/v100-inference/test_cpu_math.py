"""Run in the existing CPU image with the unmodified, pinned upstream Python source as input."""


def main():
    import ast
    import hashlib
    from pathlib import Path
    import sys
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from infer import greedy

    assert not list(Path("/dev").glob("nvidia*")) and not torch.cuda.is_initialized()
    if len(sys.argv) == 1:
        from transformers.models.qwen3_5 import modeling_qwen3_5
        functions = vars(modeling_qwen3_5)
    else:
        source = Path(sys.argv[1]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == "788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263"
        wanted = {"l2norm","torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule","causal_conv1d_fn","causal_conv1d_update"}
        nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        assert {n.name for n in nodes} == wanted
        for node in nodes:
            node.decorator_list = []  # Test the native function bodies directly, without optional dispatch.
        functions = dict(torch=torch,F=F,nn=torch.nn,ACT2FN={"silu":F.silu})
        exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned_transformers_native_fallback","exec"),functions)
    torch.set_num_threads(2)
    gen = torch.Generator(device="cpu").manual_seed(42)
    def rand(shape):
        return torch.randn(shape, generator=gen).half() * 0.1

    with torch.inference_mode():
        # Full-size value heads and dimensions, chunk boundary and one cached token.
        q,k,v = (rand((1,66,48,128)) for _ in range(3))
        g,beta = -torch.rand((1,66,48),generator=gen),torch.rand((1,66,48),generator=gen).half()
        rule = functions["torch_recurrent_gated_delta_rule"]
        options = dict(initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
        chunk, state = functions["torch_chunk_gated_delta_rule"](*(x[:,:65] for x in (q,k,v,g,beta)), **options)
        tail, end = rule(*(x[:,65:] for x in (q,k,v,g,beta)), **dict(options,initial_state=state))
        ref, refstate = rule(q,k,v,g,beta,**options)
        assert state.shape == (1,48,128,128) and state.dtype == end.dtype == torch.float32
        assert chunk.dtype == tail.dtype == torch.float16
        torch.testing.assert_close(torch.cat((chunk,tail),1),ref,atol=1e-4,rtol=1e-2)
        # Compare the two reduction orders at the FP16-input precision of this fixture.
        diff = (end-refstate).float()
        relative_l2 = (diff.norm()/refstate.norm()).item()
        print(f"FP32 state max_abs={diff.abs().max().item():.8g} relative_l2={relative_l2:.8g}")
        assert end.isfinite().all() and relative_l2 < torch.finfo(torch.float16).eps
        torch.testing.assert_close(end,refstate,atol=5e-5,rtol=1e-3)
        x,w = rand((1,10240,9)),rand((10240,4))
        conv = functions["causal_conv1d_fn"](x,w,activation="silu")
        step = functions["causal_conv1d_update"](x[:,:,-1:],x[:,:,-5:-1].clone(),w,activation="silu")
        torch.testing.assert_close(conv[:,:,-1:],step)
        with sdpa_kernel([SDPBackend.MATH]):
            a = F.scaled_dot_product_attention(rand((1,24,256,256)),rand((1,4,256,256)),
                             rand((1,4,256,256)),enable_gqa=True,is_causal=True)
            assert a.shape == (1,24,256,256) and a.dtype==torch.float16 and a.isfinite().all()
        class Toy:
            config = SimpleNamespace(vocab_size=8)
            def __init__(self, invalid=None):
                self.invalid = invalid
            def __call__(self, input_ids, past_key_values, use_cache, logits_to_keep):
                step = past_key_values or 0
                scores = torch.zeros((1,1,8))
                scores[0,0,step+1] = 1
                if step == 1 and self.invalid is not None:
                    scores[0,0,-1] = self.invalid
                return SimpleNamespace(logits=scores,past_key_values=step+1)
        ids = torch.tensor([[2,3]])
        assert greedy(Toy(),ids,{2})[0]==[1,2]
        from threading import Event
        cancelled = Event()
        observed = []

        def first_token(token):
            observed.append(token)
            cancelled.set()

        completed, cache, _, _ = greedy(Toy(), ids, {2}, on_token=first_token, cancelled=cancelled)
        assert completed == observed == [1] and cache == 1
        completed, cache, _, _ = greedy(Toy(), ids, {2}, on_token=first_token, cancelled=cancelled)
        assert completed == [] and cache is None and observed == [1]
        for invalid in (float("nan"),float("-inf"),float("inf")):
            try:
                greedy(Toy(invalid),ids,{2})
            except RuntimeError as e:
                assert "Nonfinite raw logits" in str(e)
            else:
                raise AssertionError("Accepted a nonfinite logit in cached decode")
    print("PASS: native chunk/recurrent agreement, FP32 state, FP16 convolution and math GQA, greedy EOS and rejection of every tested nonfinite raw logit")


if __name__ == "__main__":
    main()
