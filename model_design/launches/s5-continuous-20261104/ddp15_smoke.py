import argparse
import datetime
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muse-config", action="store_true")
    parser.add_argument("--s5-wrapper", action="store_true")
    parser.add_argument("--skip-optimizer", action="store_true")
    parser.add_argument("--with-data", action="store_true")
    parser.add_argument("--close-data", action="store_true")
    parser.add_argument("--barrier", action="store_true")
    parser.add_argument("--checkpoint", default="model_design/runs/s5-continuous-four-domain-v1/smoke/ddp7_attempt6/checkpoints/step_00000000.pt")
    parser.add_argument("--parent", default="model_design/runs/stage2-dbaasp-expanded-300/checkpoints/final.pt")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))
    device = torch.device(f"cuda:{local_rank}")
    data = None
    if args.with_data:
        from muse_encoder.config import MuseConfig
        from muse_encoder.model import MuseEncoder
        from muse_encoder.s5.data import S5Data
        from muse_encoder.s5.runtime import load_yaml
        from muse_encoder.tokenizers import load_tokenizers
        config = load_yaml("model_design/configs/s5_smoke.yaml")
        cfg = MuseConfig(**config["model"])
        data = S5Data(config, cfg, load_tokenizers())
        cfg.aux_specs = data.aux_specs
        if args.barrier:
            print(f"[ddp-smoke] rank={rank} data-barrier-start", flush=True)
            dist.barrier()
            print(f"[ddp-smoke] rank={rank} data-barrier-done", flush=True)
        model = MuseEncoder(cfg).to(device)
        if args.close_data:
            data.close()
            data = None
    elif args.muse_config:
        from muse_encoder.config import MuseConfig
        from muse_encoder.model import MuseEncoder
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = MuseEncoder(MuseConfig(**payload["config"]["model"])).to(device)
    else:
        model = nn.Linear(1024, 1024).to(device)
    if args.s5_wrapper:
        from muse_encoder.pretrain.trainer import seed_everything
        from muse_encoder.s5.runtime import S5Wrapper, build_optimizer, strict_initialize
        seed_everything(42, True)
        init_record = strict_initialize(model, args.parent, rank)
        model = S5Wrapper(model, {})
        if not args.skip_optimizer:
            build_optimizer(model, init_record, 1e-5, 1e-4, 0.01)
        print(f"[ddp-smoke] rank={rank} s5-wrapper-ready", flush=True)
    print(f"[ddp-smoke] rank={rank} before-ddp", flush=True)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    print(f"[ddp-smoke] rank={rank} after-ddp", flush=True)
    if not args.muse_config and not args.with_data:
        output = model(torch.randn(8, 1024, device=device)).sum()
        output.backward()
        torch.cuda.synchronize()
        print(f"[ddp-smoke] rank={rank} backward-ok loss={float(output)}", flush=True)
    if data is not None:
        data.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
