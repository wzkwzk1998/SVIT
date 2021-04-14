import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout =0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Model(nn.Module):
    def __init__(self, *, temporal_size, temporal_stride, num_joint, num_classes, dim,
                 depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0.5, emb_dropout=0.5):
        super().__init__()
        assert temporal_size % temporal_stride == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (temporal_size // temporal_stride)
        patch_dim = temporal_stride * num_joint * channels
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('n c t v m -> (n m) t v c'),
            Rearrange('b (np ts) v c -> b np (ts v c)', np = num_patches, ts = temporal_stride),

            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = rearrange(x, '(b m) n d -> b m n d', m=2)
        x = x.mean(dim=1)                 # mean two body frame
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# class Model(nn.Module):
#     def __init__(self, *, temporal_size, temporal_stride, num_joint, num_classes, dim,
#                  depth, heads, mlp_dim, pool='cls', channels=3,
#                  dim_head=64, dropout=0.5, emb_dropout=0.5):
#         super().__init__()
#         assert temporal_size % temporal_stride == 0, 'Image dimensions must be divisible by the patch size.'
#         num_patches = (temporal_size // temporal_stride)
#         spatial_patch_dim = temporal_stride * channels
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('n c t v m -> (n m) t v c'),
#             Rearrange('b (np ts) v c -> b np v (ts c)', np = num_patches, ts = temporal_stride),
#             # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
#             nn.Linear(spatial_patch_dim, dim),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, num_joint+1, dim))
#         self.space_cls_token = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.temporal_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.to_patch_embedding(x)
#         b, t, n, _ = x.shape
#
#         cls_space_tokens = repeat(self.space_cls_token, '() n d -> b t n d', b = b, t  = t)
#         x = torch.cat((cls_space_tokens, x), dim=2)
#         x += self.pos_embedding[:, :, :(n + 1)]
#         x = self.dropout(x)
#
#         x = rearrange(x, 'b t n d -> (b t) n d')
#         x = self.spatial_transformer(x)
#         x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b, t=t)
#
#         cls_temporal_tokens = repeat(self.temporal_cls_token, '() n d -> b n d', b=b)
#         x = torch.cat((cls_temporal_tokens, x), dim=1)
#
#         x = self.temporal_transformer(x)
#
#         x = rearrange(x, '(b m) n d -> b m n d', m=2)
#         x = x.mean(dim=1)                 # mean two body frame
#         x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
#
#         x = self.to_latent(x)
#         return self.mlp_head(x)
