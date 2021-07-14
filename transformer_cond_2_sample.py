#!/usr/bin/env python3

"""Samples an image from a CLIP conditioned Decision Transformer."""

import argparse
from pathlib import Path
import sys

from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import top_k_top_p_filtering
from tqdm import trange

sys.path.append('./taming-transformers')

from CLIP import clip
from taming.models import vqgan


def setup_exceptions():
    try:
        from IPython.core.ultratb import FormattedTB
        sys.excepthook = FormattedTB(mode='Plain', color_scheme='Neutral')
    except ImportError:
        pass


class CausalTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None, cache=None):
        output = src

        if self.training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            if self.norm is not None:
                output = self.norm(output)

            return output

        new_token_cache = []
        compute_len = src.shape[0]
        if cache is not None:
            compute_len -= cache.shape[1]
        for i, mod in enumerate(self.layers):
            output = mod(output, compute_len=compute_len)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CausalTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, compute_len=None):
        if self.training:
            return super().forward(src, src_mask, src_key_padding_mask)

        if compute_len is None:
            src_last_tok = src
        else:
            src_last_tok = src[-compute_len:, :, :]

        attn_mask = src_mask if compute_len > 1 else None
        tmp_src = self.self_attn(src_last_tok, src, src, attn_mask=attn_mask,
                                 key_padding_mask=src_key_padding_mask)[0]
        src_last_tok = src_last_tok + self.dropout1(tmp_src)
        src_last_tok = self.norm1(src_last_tok)

        tmp_src = self.linear2(self.dropout(self.activation(self.linear1(src_last_tok))))
        src_last_tok = src_last_tok + self.dropout2(tmp_src)
        src_last_tok = self.norm2(src_last_tok)
        return src_last_tok


class CLIPToImageTransformer(nn.Module):
    def __init__(self, clip_dim, seq_len, n_toks):
        super().__init__()
        self.clip_dim = clip_dim
        d_model = 1024
        self.clip_in_proj = nn.Linear(clip_dim, d_model, bias=False)
        self.clip_score_in_proj = nn.Linear(1, d_model, bias=False)
        self.in_embed = nn.Embedding(n_toks, d_model)
        self.out_proj = nn.Linear(d_model, n_toks)
        layer = CausalTransformerEncoderLayer(d_model, d_model // 64, d_model * 4,
                                              dropout=0, activation='gelu')
        self.encoder = CausalTransformerEncoder(layer, 24)
        self.pos_emb = nn.Parameter(torch.zeros([seq_len + 1, d_model]))
        self.register_buffer('mask', self._generate_causal_mask(seq_len + 1), persistent=False)

    @staticmethod
    def _generate_causal_mask(size):
        mask = (torch.triu(torch.ones([size, size])) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        mask[0, 1] = 0
        return mask

    def forward(self, clip_embed, clip_score, input=None, cache=None):
        if input is None:
            input = torch.zeros([len(clip_embed), 0], dtype=torch.long, device=clip_embed.device)
        clip_embed_proj = self.clip_in_proj(F.normalize(clip_embed, dim=1) * self.clip_dim**0.5)
        clip_score_proj = self.clip_score_in_proj(clip_score)
        embed = torch.cat([clip_embed_proj.unsqueeze(0),
                           clip_score_proj.unsqueeze(0),
                           self.in_embed(input.T)])
        embed_plus_pos = embed + self.pos_emb[:len(embed)].unsqueeze(1)
        mask = self.mask[:len(embed), :len(embed)]
        out, cache = self.encoder(embed_plus_pos, mask, cache=cache)
        return self.out_proj(out[1:]).transpose(0, 1), cache


def main():
    setup_exceptions()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('prompt', type=str,
                   help='the prompt')
    p.add_argument('--batch-size', '-bs', type=int, default=4,
                   help='the batch size')
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='the checkpoint to use')
    p.add_argument('--clip-score', type=float, default=1.,
                   help='the CLIP score to condition on')
    p.add_argument('--device', type=str, default=None,
                   help='the device to use')
    p.add_argument('--half', action='store_true',
                   help='use half precision')
    p.add_argument('-k', type=int, default=1,
                   help='the number of samples to save')
    p.add_argument('-n', type=int, default=1,
                   help='the number of samples to draw')
    p.add_argument('--output', '-o', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    p.add_argument('--temperature', type=float, default=1.,
                   help='the softmax temperature for sampling')
    p.add_argument('--top-k', type=int, default=0,
                   help='the top-k value for sampling')
    p.add_argument('--top-p', type=float, default=1.,
                   help='the top-p value for sampling')
    p.add_argument('--vqgan-checkpoint', type=Path, required=True,
                   help='the VQGAN checkpoint (.ckpt)')
    p.add_argument('--vqgan-config', type=Path, required=True,
                   help='the VQGAN config (.yaml)')
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dtype = torch.half if args.half else torch.float

    perceptor = clip.load('ViT-B/32', jit=False)[0].to(device).eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    vqgan_config = OmegaConf.load(args.vqgan_config)
    vqgan_model = vqgan.VQModel(**vqgan_config.model.params).to(device)
    vqgan_model.eval().requires_grad_(False)
    vqgan_model.init_from_ckpt(args.vqgan_checkpoint)
    del vqgan_model.loss

    clip_dim = perceptor.visual.output_dim
    clip_input_res = perceptor.visual.input_resolution
    e_dim = vqgan_model.quantize.e_dim
    f = 2**(vqgan_model.decoder.num_resolutions - 1)
    n_toks = vqgan_model.quantize.n_e
    size_x, size_y = 384, 384
    toks_x, toks_y = size_x // f, size_y // f

    torch.manual_seed(args.seed)

    text_embed = perceptor.encode_text(clip.tokenize(args.prompt).to(device)).to(dtype)
    text_embed = text_embed.repeat([args.n, 1])
    clip_score = torch.ones([text_embed.shape[0], 1], device=device, dtype=dtype) * args.clip_score

    model = CLIPToImageTransformer(clip_dim, toks_y * toks_x, n_toks)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model = model.to(device, dtype).eval().requires_grad_(False)

    @torch.no_grad()
    def sample(clip_embed, clip_score, temperature=1., top_k=0, top_p=1.):
        tokens = torch.zeros([len(clip_embed), 0], dtype=torch.long, device=device)
        cache = None
        for i in trange(toks_y * toks_x, leave=False):
            logits, cache = model(clip_embed, clip_score, tokens, cache=cache)
            logits = logits[:, -1] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            next_token = logits.softmax(1).multinomial(1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    def decode(tokens):
        z = vqgan_model.quantize.embedding(tokens).view([-1, toks_y, toks_x, e_dim]).movedim(3, 1)
        return vqgan_model.decode(z).add(1).div(2).clamp(0, 1)

    try:
        out_lst, sim_lst = [], []
        for i in trange(0, len(text_embed), args.batch_size):
            tokens = sample(text_embed[i:i+args.batch_size], clip_score[i:i+args.batch_size],
                            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            out = decode(tokens)
            out_lst.append(out)
            out_for_clip = F.interpolate(out, (clip_input_res, clip_input_res),
                                         mode='bilinear', align_corners=False)
            image_embed = perceptor.encode_image(normalize(out_for_clip)).to(dtype)
            sim = torch.cosine_similarity(text_embed[i:i+args.batch_size], image_embed)
            sim_lst.append(sim)
        out = torch.cat(out_lst)
        sim = torch.cat(sim_lst)
        best_values, best_indices = sim.topk(min(args.k, args.n))
        for i, index in enumerate(best_indices):
            TF.to_pil_image(out[index]).save(args.output + f'_{i:03}.png')
            print(f'Actual CLIP score for output {i}: {best_values[i].item():g}')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
