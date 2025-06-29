# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    frame_sample_rate=2,
    deterministic=True,
    log_dir=None,
    property=None,
):
    from evals.intphys2.intphys2_dataset import make_videodataset
    print(root_path)
    dataset, data_loader, dist_sampler = make_videodataset(
        data_path=root_path,
        batch_size=batch_size,
        frame_step=frame_sample_rate,
        shared_transform=shared_transform,
        transform=transform,
        collator=collator,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        deterministic=deterministic,
        log_dir=log_dir,
        pin_mem=pin_mem,
        drop_last=False)

    return (data_loader, dist_sampler)
