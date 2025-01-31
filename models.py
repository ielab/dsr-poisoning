from torch import nn
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor
from typing import List, Optional, Union
from utils import to_channel_dimension_format
from tqdm import tqdm

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize



class Model(nn.Module):
    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        self.mean = self.mean.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)
        return res

    def get_query_embeddings(self, query_texts: List[str], batch_size: int) -> torch.Tensor:
        raise NotImplementedError

    def get_doc_embeddings(self, doc_images: List[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    def get_doc_inputs(self, doc_images: List[Image.Image]):
        raise NotImplementedError

    def get_doc_embedding_by_tensor(self, image, # RGB 3 channel
                                    doc_inputs,
                                    resized_height,
                                    resized_width,
                                    input_data_format,
                                    ) -> torch.Tensor:
        # this function is torch reimplementing image processor which uses numpy. Make sure gradient can flow.
        raise NotImplementedError

    def compute_similarity(self, query_embeddings, doc_embeddings):
        raise NotImplementedError




class DSE(Model):
    def __init__(self, model_name="MrLight/dse-qwen2-2b-mrl-v1", cache_dir=None, disable_flash_attention=True):
        super(DSE, self).__init__()
        self.min_pixels = 1 * 28 * 28
        self.max_pixels = 2560 * 28 * 28
        self.patch_size = 14
        self.merge_size = 2
        self.rescale_factor = 1 / 255
        self.temporal_patch_size = 2
        self.mean = torch.Tensor(OPENAI_CLIP_MEAN)
        self.std = torch.Tensor(OPENAI_CLIP_STD)
        self.processor = AutoProcessor.from_pretrained(model_name,
                                                  min_pixels=self.min_pixels,
                                                  max_pixels=self.max_pixels,
                                                  cache_dir=cache_dir)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name,
                                                                attn_implementation=None if disable_flash_attention else "flash_attention_2",
                                                                torch_dtype=torch.bfloat16,
                                                                cache_dir=cache_dir).eval()

        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"


    def _get_embedding(self, last_hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps[:, :dimension], p=2, dim=-1)
        return reps

    def get_query_embeddings(self, query_texts: List[str], batch_size=64) -> torch.Tensor:
        query_messages = []
        for query in query_texts:
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': Image.new('RGB', (28, 28)), 'resized_height': 1, 'resized_width': 1},
                        # need a dummy image here for an easier process.
                        {'type': 'text', 'text': f'Query: {query}'},
                    ]
                }
            ]
            query_messages.append(message)

        query_embeddings = []
        for i in tqdm(range(0, len(query_messages), batch_size), desc='Encoding queries'):
            query_texts_batch = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
                for msg in query_messages[i : i + batch_size]
            ]

            query_image_inputs, query_video_inputs = process_vision_info(query_messages[i : i + batch_size])
            query_inputs = self.processor(text=query_texts_batch, images=query_image_inputs, videos=query_video_inputs,
                                        padding='longest', return_tensors='pt').to(self.model.device)
            cache_position = torch.arange(query_inputs['input_ids'].shape[1]).to(self.model.device)
            query_inputs = self.model.prepare_inputs_for_generation(**query_inputs, cache_position=cache_position,
                                                                use_cache=False)
            with torch.no_grad():
                output = self.model(**query_inputs, return_dict=True, output_hidden_states=True)
            query_embeddings.append(self._get_embedding(output.hidden_states[-1], 1536))  # adjust dimensionality for efficiency trade-off, e.g. 512

        return torch.cat(query_embeddings, dim=0)


    def get_doc_inputs(self, doc_images: List[Image.Image]):
        doc_messages = []
        for doc in doc_images:
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': doc, },
                        # 'resized_height':680 , 'resized_width':680} # adjust the image size for efficiency trade-off
                        {'type': 'text', 'text': 'What is shown in this image?'}
                    ]
                }
            ]
            doc_messages.append(message)
        doc_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
            for msg in doc_messages
        ]
        doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
        doc_inputs = self.processor(text=doc_texts, images=doc_image_inputs, videos=doc_video_inputs, padding='longest',
                               return_tensors='pt').to(self.model.device)
        cache_position = torch.arange(doc_inputs['input_ids'].shape[1]).to(self.model.device)
        doc_inputs = self.model.prepare_inputs_for_generation(**doc_inputs, cache_position=cache_position, use_cache=False)

        height, width = doc_image_inputs[0].size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return doc_inputs, resized_height, resized_width


    def get_doc_embeddings(self, doc_images: List[Image.Image]) -> torch.Tensor:
        doc_inputs, _, _ = self.get_doc_inputs(doc_images)
        output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
        doc_embeddings = self._get_embedding(output.hidden_states[-1],
                                       1536)  # adjust dimensionality for efficiency trade-off e.g. 512
        return doc_embeddings


    def get_doc_embedding_by_tensor(self, image,
                                    doc_inputs,
                                    resized_height,
                                    resized_width,
                                    input_data_format,
                                    ) -> torch.Tensor:
        image_norm = image * self.rescale_factor
        image_norm = (image_norm - self.mean) / self.std

        patches = torch.stack([image_norm])
        if input_data_format == ChannelDimension.LAST:
            patches = patches.permute(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = patches.repeat(self.temporal_patch_size, 1, 1, 1)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        doc_inputs['pixel_values'] = flatten_patches

        # Forward pass
        output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
        doc_embeddings = self._get_embedding(output.hidden_states[-1], 1536)
        return doc_embeddings

    def compute_similarity(self, query_embeddings, doc_embeddings):
        if isinstance(doc_embeddings, List):
            doc_embeddings = torch.stack(doc_embeddings)

        return torch.mm(query_embeddings, doc_embeddings.T)



class ColPali(Model):
    def __init__(self, model_name="vidore/colpali-v1.2-hf", cache_dir=None):
        super(ColPali, self).__init__()
        self.model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir).eval()

        self.processor = ColPaliProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.rescale_factor = self.processor.image_processor.rescale_factor
        self.mean = torch.Tensor(self.processor.image_processor.image_mean)
        self.std = torch.Tensor(self.processor.image_processor.image_std)

    def get_query_embeddings(self, query_texts: List[str], batch_size=64, max_length=128) -> torch.Tensor:
        query_embeddings = []
        for i in tqdm(range(0, len(query_texts), batch_size), desc='Encoding queries'):
            query_texts_batch = query_texts[i : i + batch_size]
            batch_queries = self.processor(text=query_texts_batch,
                                           padding='max_length',
                                           max_length=max_length,
                                           return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                query_embeddings.append(self.model(**batch_queries).embeddings)
        return torch.cat(query_embeddings, dim=0)

        # batch_queries = self.processor(text=query_texts).to(self.model.device)
        # # Forward pass
        # with torch.no_grad():
        #     query_embeddings = self.model(**batch_queries).embeddings
        # return query_embeddings

    def get_doc_embeddings(self, doc_images: List[Image.Image]) -> torch.Tensor:
        # Process the inputs
        batch_images = self.processor(images=doc_images).to(self.model.device)
        # Forward pass
        with torch.no_grad():
            image_embeddings = self.model(**batch_images).embeddings
        return image_embeddings


    def get_doc_inputs(self, doc_images: List[Image.Image]):
        doc_inputs = self.processor(images=doc_images).to(self.model.device)
        return doc_inputs, self.processor.image_processor.size['height'], self.processor.image_processor.size['width']

    def get_doc_embedding_by_tensor(self, image, # 3D, (H, W, C)
                                    doc_inputs,
                                    resized_height,
                                    resized_width,
                                    input_data_format,
                                    ) -> torch.Tensor:

        image_norm = image * self.rescale_factor
        image_norm = (image_norm - self.mean) / self.std
        pixel_values = to_channel_dimension_format(image_norm, ChannelDimension.FIRST,
                                                 input_channel_dim=input_data_format).unsqueeze(0)
        doc_inputs['pixel_values'] = pixel_values

        # Forward pass
        output = self.model(**doc_inputs, return_dict=True)
        doc_embeddings = output.embeddings
        return doc_embeddings

    def compute_similarity(self, query_embeddings, doc_embeddings):
        return self.processor.score_retrieval(query_embeddings, doc_embeddings)