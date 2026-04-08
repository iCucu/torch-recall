import torch
import torch._inductor

from torch_recall.schema import MAX_BP, MAX_NP, P_TOTAL, CONJ_PER_PASS
from torch_recall.model import InvertedIndexModel


def create_example_inputs(model: InvertedIndexModel, device: str = "cpu") -> tuple:
    return (
        torch.zeros(MAX_BP, dtype=torch.int64, device=device),
        torch.zeros(MAX_BP, dtype=torch.bool, device=device),
        torch.zeros(MAX_NP, dtype=torch.int64, device=device),
        torch.zeros(MAX_NP, dtype=torch.int64, device=device),
        torch.zeros(MAX_NP, dtype=torch.float32, device=device),
        torch.zeros(MAX_NP, dtype=torch.bool, device=device),
        torch.zeros(P_TOTAL, dtype=torch.bool, device=device),
        torch.zeros(CONJ_PER_PASS, P_TOTAL, dtype=torch.bool, device=device),
        torch.zeros(CONJ_PER_PASS, dtype=torch.bool, device=device),
    )


def export_model(model: InvertedIndexModel, meta: dict, output_path: str) -> str:
    model.eval()
    device = next(model.buffers()).device

    example_inputs = create_example_inputs(model, device=str(device))

    with torch.no_grad():
        exported = torch.export.export(model, example_inputs)

    path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
    )
    return path
