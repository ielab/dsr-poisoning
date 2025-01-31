import torch
from models import DSE, ColPali
from PIL import Image
import numpy as np
from transformers.image_utils import (
    to_numpy_array,
    infer_channel_dimension_format,
    PILImageResampling,
)
from transformers.image_transforms import (
    resize,
    convert_to_rgb
)
from utils import add_margin
import json
import os
from argparse import ArgumentParser


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True


def main(args):
    mask_ratio = args.mask_ratio
    optimization_method = args.optimization_method
    init_noise = args.init_noise
    grad_ratio = args.grad_ratio
    cache_dir = args.cache_dir
    seed_image = args.seed_image
    query_file = f'{args.target_query_file}'
    optimize_steps = args.num_steps
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.model == 'dse':
        model = DSE(cache_dir=cache_dir).to('cuda').eval()
    elif args.model == 'colpali':
        model = ColPali(cache_dir=cache_dir).to('cuda').eval()
    else:
        raise ValueError('Model not supported')

    adv_image = Image.open(seed_image)
    adv_image = convert_to_rgb(adv_image)
    adv_image = adv_image.resize((512, 512))

    # prepare doc inputs
    doc_inputs, resized_height, resized_width = model.get_doc_inputs([adv_image])

    # Scale to be between 0 and 1
    adv_image_np = to_numpy_array(adv_image)
    input_data_format = infer_channel_dimension_format(adv_image_np)
    adv_image_np = resize(
        adv_image_np, size=(resized_height, resized_width),
        resample=PILImageResampling.BICUBIC,
        input_data_format=input_data_format
    )
    original_image_pt = torch.clone(torch.tensor(adv_image_np, dtype=torch.float32))
    original_image_pt = original_image_pt.to(model.model.device)


    if mask_ratio > 0:
        original_image_pt, grad_mask = add_margin(original_image_pt, mask_ratio, init_black=False)
    else:
        grad_mask = torch.ones_like(original_image_pt, dtype=torch.bool)

    with open(query_file, 'r') as f:
        query_clusters = json.load(f)

    for k in query_clusters:
        # get targe query embeddings
        queries = query_clusters[k]
        print("Num of queries", len(queries))
        with torch.no_grad():
            query_embeddings = model.get_query_embeddings(queries).detach()
        query_embeddings = query_embeddings.to(model.model.device)
        adv_image_pt = torch.tensor(adv_image_np, dtype=torch.float32)

        if mask_ratio > 0:
            adv_image_pt, grad_mask = add_margin(adv_image_pt, mask_ratio, init_black=False)

        # initialize noise
        if optimization_method == "noise":
            if init_noise == "zeros":
                noise = torch.zeros_like(original_image_pt, requires_grad=True)
            elif init_noise == "uniform":
                noise = torch.randn_like(original_image_pt, requires_grad=True) * 255
            elif init_noise == "ones":
                noise = torch.ones_like(original_image_pt, requires_grad=True) * 255
            else:
                raise ValueError("Noise initialization not supported")

            noise = noise * grad_mask.to(noise.device)
            noise = noise.to(model.model.device)

        elif optimization_method == "direct":
            adv_image_pt = torch.tensor(adv_image_pt, requires_grad=True, dtype=torch.float32)
            adv_image_pt.data = adv_image_pt.data.to(model.model.device)
        else:
            raise ValueError("Optimization method not supported")

        # adversarial attack
        optimizer = torch.optim.Adam(model.model.parameters(), lr=1.0) # used just for the scheduler, we manually update the image
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_steps, eta_min=0.0)

        grad_mask = grad_mask.to(model.model.device)

        for step in range(optimize_steps):  # num_steps is the number of optimization iterations

            if optimization_method == "direct":
                adv_image_pt = torch.tensor(adv_image_pt, requires_grad=True, dtype=torch.float32)
                adv_image_pt = adv_image_pt.to(model.model.device)
                adv_image_with_noise = adv_image_pt

            elif optimization_method == "noise":
                noise = torch.tensor(noise, requires_grad=True, dtype=torch.float32)
                noise = noise.to(model.model.device)
                adv_image_pt = adv_image_pt.to(model.model.device)

                # Recompute pixel values tensor from images_to_update
                adv_image_with_noise = adv_image_pt + noise
                adv_image_with_noise = adv_image_with_noise.clamp(0, 255)

            # # Get the doc embeddings
            doc_embeddings = model.get_doc_embedding_by_tensor(adv_image_with_noise,
                                                               doc_inputs,
                                                               resized_height,
                                                               resized_width,
                                                               input_data_format=input_data_format)

            # Compute the loss
            loss = model.compute_similarity(query_embeddings, doc_embeddings)
            loss = -loss.mean()

            # Backpropagation
            loss.backward()

            if optimization_method == "direct":
                # Get the gradient
                data_grad = adv_image_pt.grad.data
            elif optimization_method == "noise":
                data_grad = noise.grad.data

            # only update the gradient where the mask is true
            data_grad = data_grad * grad_mask

            if grad_ratio is not None:
                data_grad = torch.where(torch.abs(data_grad) > torch.quantile(torch.abs(data_grad), 1 - grad_ratio),
                                        data_grad, torch.zeros_like(data_grad))
            data_grad = data_grad / torch.norm(data_grad, p=1)

            lr = lr_scheduler.get_lr()[0]
            lr_scheduler.step()

            # Update the image
            if optimization_method == "direct":
                adv_image_pt = adv_image_pt - lr * torch.sign(data_grad)
                adv_image_pt = torch.clamp(adv_image_pt, 0, 255)
            elif optimization_method == "noise":
                noise = noise - lr * torch.sign(data_grad)
                noise = torch.clamp(noise, 0, 255)

            if step % 10 == 0:
            #    # Print loss for debugging
                print(f"Step {step}: {loss.item()}")

        ndarr = adv_image_with_noise.to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(f'{save_dir}/adv_img_{k}.png')


if __name__ == '__main__':
    # read arguments
    parser = ArgumentParser()
    parser.add_argument('--seed_image', type=str, required=True, help='Path to seed document')
    parser.add_argument('--target_query_file', type=str, required=True, help='Path to seed queries')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to dir for saving adversarial images.')
    parser.add_argument('--model', type=str, required=True, help='model name. Either dse or colpali')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory to store the model')
    parser.add_argument('--num_steps', type=int, default=3000, help='number of optimization steps')
    parser.add_argument('--mask_ratio', type=float, default=0.0, help='mask ratio')
    parser.add_argument('--optimization_method', type=str, default="direct", help='Optimization method. Possible values: direct, noise')
    parser.add_argument('--init_noise', type=str, default="zeros", help='Noise initialization. Possible values: zeros, uniform, ones')
    parser.add_argument('--grad_ratio', type=float, default=None, help='percentage of gradient to keep, between 0 and 1')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.random_seed)
    main(args)
