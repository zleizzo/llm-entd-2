"""Train with a fixed-window loader driven by a persisted schedule (window order per step).

Usage: python -m entd.train configs/x.yaml
DDP:   torchrun --nproc_per_node 8 -m entd.train configs/x.yaml   (each rank takes batch[rank::world])
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

from .config import load_config
from .data import ensure_schedule, ensure_tables, load_stream, prepare
from .model import build_model
from .util import autocast_ctx, gather_windows, pick_device


def save_atomic(obj, path):
    """Write to path.tmp then rename, so a concurrent reader never sees a partial checkpoint."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def get_lr(step, total, cfg):
    t = cfg["train"]
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    if total <= t["warmup_steps"]:
        return t["min_lr"]
    r = (step - t["warmup_steps"]) / max(1, total - t["warmup_steps"])
    return t["min_lr"] + 0.5 * (1 + math.cos(math.pi * min(1.0, r))) * (t["lr"] - t["min_lr"])


@torch.no_grad()
def estimate_loss(model, stream, offsets, T, n_batches, bs, device, dtype):
    model.eval()
    losses = []
    for b in range(n_batches):
        offs = offsets[b * bs:(b + 1) * bs]
        if len(offs) == 0:
            break
        tok = torch.from_numpy(gather_windows(stream, offs, T)).to(device)
        with autocast_ctx(device, dtype):
            _, loss, _ = model(tok[:, :T], tok[:, 1:T + 1])
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main(cfg_path):
    cfg = load_config(cfg_path)
    meta = prepare(cfg)
    tables = ensure_tables(cfg, meta)
    sched, _ = ensure_schedule(cfg, tables)
    T = tables["T"]
    tc = cfg["train"]

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = pick_device(tc["device"])
    master = rank == 0
    torch.manual_seed(tc["seed"] + rank)

    train_stream = load_stream(cfg["data"]["dir"], "train")
    val_stream = load_stream(cfg["data"]["dir"], "val")
    W_tr, W_va = tables["windows_train"], tables["windows_val"]
    trained = np.nonzero(tables["split_train"] == 0)[0]

    model = build_model(cfg, meta, device)
    if master:
        print(f"[train] device={device} params={model.num_params()/1e6:.2f}M (non-pos-emb) "
              f"windows={len(W_tr)} trained={len(trained)} steps={len(sched)} T={T}", flush=True)
    opt = model.configure_optimizers(tc["weight_decay"], tc["lr"], (tc["beta1"], tc["beta2"]), device.type)
    if tc["compile"] and device.type == "cuda":
        model = torch.compile(model)
    raw = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    ck_path = os.path.join(cfg["work_dir"], "ckpt.pt")
    start = 0
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu")
        if ck.get("step", 0) >= len(sched):
            if master:
                print("[train] checkpoint already complete", flush=True)
            return
        raw.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"]
        if master:
            print(f"[train] resuming from step {start}", flush=True)

    log = open(os.path.join(cfg["work_dir"], "train_log.jsonl"), "a") if master else None
    eval_offs_val = W_va[: tc["eval_batches"] * tc["batch_size"]]
    eval_offs_tr = W_tr[trained[: tc["eval_batches"] * tc["batch_size"]]]
    total = len(sched)
    steps_per_epoch = total // max(1, tc["epochs"])   # make_schedule gives exactly epochs * n_b steps
    plain_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    t0 = time.time()
    model.train()
    for step in range(start, total):
        lr = get_lr(step, total, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        wids = sched[step][rank::world]
        tok = torch.from_numpy(gather_windows(train_stream, W_tr[wids], T)).to(device)
        with autocast_ctx(device, tc["dtype"]):
            _, loss, _ = model(tok[:, :T], tok[:, 1:T + 1])
        loss.backward()
        if tc["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = step == total - 1
        if master and (step % tc["log_interval"] == 0 or last):
            el = time.time() - t0
            print(f"step {step}/{total} loss {loss.item():.4f} lr {lr:.2e} {el:.0f}s", flush=True)
        if master and (step % tc["eval_interval"] == 0 or last) and (step > start or last):
            vl = estimate_loss(raw, val_stream, eval_offs_val, T, tc["eval_batches"], tc["batch_size"], device, tc["dtype"])
            tl = estimate_loss(raw, train_stream, eval_offs_tr, T, tc["eval_batches"], tc["batch_size"], device, tc["dtype"])
            rec = {"step": step, "train_loss": tl, "val_loss": vl, "lr": lr, "time": time.time() - t0}
            print(f"[eval] {rec}", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            save_atomic({"model": raw.state_dict(), "optimizer": opt.state_dict(), "step": step + 1,
                         "model_args": raw.config.__dict__, "cfg": plain_cfg}, ck_path)
        if master and steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0:
            # Model-only snapshot at the end of each epoch (ckpt_epoch1.pt, ...); point another config's
            # model.checkpoint at it to embed/eval an earlier model with the same data/split.
            ep = (step + 1) // steps_per_epoch
            save_atomic({"model": raw.state_dict(), "step": step + 1, "epoch": ep,
                         "model_args": raw.config.__dict__, "cfg": plain_cfg},
                        os.path.join(cfg["work_dir"], f"ckpt_epoch{ep}.pt"))
            print(f"[train] saved ckpt_epoch{ep}.pt at step {step + 1}", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
