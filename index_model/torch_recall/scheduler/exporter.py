from __future__ import annotations

import torch
import torch._inductor


def export_recall_model(
    model: torch.nn.Module,
    output_path: str,
) -> str:
    """Export any recall model to a .pt2 AOTInductor package.

    The model must have an ``example_inputs(device)`` method that returns
    a tuple of example tensors for tracing.
    """
    model.eval()

    device = "cpu"
    for buf in model.buffers():
        device = str(buf.device)
        break

    example_inputs = model.example_inputs(device=device)

    with torch.no_grad():
        exported = torch.export.export(model, example_inputs)

    path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
    )
    return path
