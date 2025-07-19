import torch


trait_masks = torch.tensor([
    [1., 1., 1., 1., 1., 1., 0., 0., 0.],
    [1., 1., 1., 1., 1., 1., 0., 0., 0.],
    [1., 1., 0., 0., 0., 0., 1., 1., 1.],
    [1., 1., 0., 0., 0., 0., 1., 1., 1.],
    [1., 1., 0., 0., 0., 0., 1., 1., 1.],
    [1., 1., 0., 0., 0., 0., 1., 1., 1.],
    [1., 1., 1., 0., 0., 1., 0., 0., 0.],
    [1., 1., 1., 1., 1., 1., 0., 0., 0.]
], dtype=torch.int8, requires_grad=False)


def get_trait_mask_by_prompt_ids(prompt_ids):
    assert type(prompt_ids) == torch.Tensor, "Only tensor type to be accept!"
    if prompt_ids.dtype not in [torch.int64, torch.bool]:
        prompt_ids = prompt_ids.to(dtype=torch.int64)
    if len(prompt_ids.size()) > 0:
        prompt_ids = prompt_ids.flatten()

    return trait_masks.to(prompt_ids.device)[prompt_ids-1]