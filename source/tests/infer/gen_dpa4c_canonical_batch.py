#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate a small compressed DPA4C canonical .pt2 API test model."""

import copy
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from gen_common import (  # noqa: E402
    ensure_inductor_compiler,
    load_custom_ops,
)

CONFIG = {
    "type_map": ["A", "B"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 3.0,
        "channels": 8,
        "lmax": 2,
        "n_radial": 8,
        "precision": "float32",
        "seed": 17,
    },
    "fitting_net": {
        "type": "ener",
        "neuron": [32, 32],
        "activation_function": "silu",
        "precision": "float32",
        "resnet_dt": False,
        "seed": 19,
    },
}


def main() -> None:
    import torch

    from deepmd.pt_expt.model.get_model import get_model
    from deepmd.pt_expt.utils.serialization import deserialize_to_file

    ensure_inductor_compiler()
    load_custom_ops()

    model = get_model(copy.deepcopy(CONFIG)).to("cpu").eval()
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() and torch.count_nonzero(parameter).item() == 0:
                parameter.copy_(
                    0.05
                    * torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        generator=generator,
                    )
                )
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)

    data = {
        "model": model.serialize(),
        "model_def_script": CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }
    path = os.path.join(
        os.path.dirname(__file__), "deeppot_dpa4c_canonical_batch.pt2"
    )
    deserialize_to_file(
        path,
        data,
        do_atomic_virial=True,
        lower_kind="dpa4c_canonical",
    )
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(
            archive.read("model/extra/metadata.json").decode("utf-8")
        )
    assert metadata["lower_input_kind"] == "dpa4c_canonical"
    assert metadata["canonical_index_dtype"] == "uint32"
    print(f"Generated {path}")  # noqa: T201


if __name__ == "__main__":
    main()
