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
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.attn_drop = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        # attn = self.attn_drop(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ST_Transformer(nn.Module):
    """this transformer is consit of spatial attention and temporal attention"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        b, t, n, d = x.shape
        for s_attn, t_attn, ff in self.layers:
            x = rearrange(x, 'b t n d -> (b t) n d')
            x = s_attn(x) + x
            x = rearrange(x, '(b t) n d -> b t n d', t=t)
            x = rearrange(x, 'b t n d -> (b n) t d')
            x = t_attn(x) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) t d -> b n t d', n=n)
            x = rearrange(x, 'b n t d-> b t n d')
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


        self.pre_bn = nn.BatchNorm1d(num_joint * channels)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('n c t v m -> (n m) t v c'),
            Rearrange('b (np ts) v c -> b np (ts v c)', np=num_patches, ts=temporal_stride),
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

    def pre_process(self, x):
        n, c, t, v, m = x.shape
        x = rearrange(x, 'n c t v m -> (n m) (v c) t')
        x = self.pre_bn(x)
        x = rearrange(x, '(n m) (v c) t -> n c t v m', m=m, v=v)
        return x

    def forward(self, x):
        x = self.pre_process(x)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = rearrange(x, '(b m) n d -> b m n d', m=2)
        x = x.mean(dim=1)  # mean two body frame
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class Model2(nn.Module):
    """
    this model is consist of separate spatial transformer and temporal transformer
    """

    def __init__(self, temporal_size, temporal_stride, num_joint, num_classes, s_dim,
                 depth, heads, s_mlp_dim, pool='cls', channels=3,
                 s_dim_head=4, dropout=0.5, emb_dropout=0.5):
        super().__init__()
        assert temporal_size % temporal_stride == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (temporal_size // temporal_stride)
        spatial_patch_dim = temporal_stride * channels
        t_dim = s_dim * num_joint
        t_dim_head = s_dim_head * num_joint
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('n c t v m -> (n m) t v c'),
            Rearrange('b (np ts) v c -> b np v (ts c)', np=num_patches, ts=temporal_stride),
            nn.Linear(spatial_patch_dim, s_dim),
        )

        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, num_joint, s_dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_transformer = Transformer(s_dim, depth, heads, s_dim_head, s_mlp_dim, dropout)

        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, t_dim))
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, t_dim))
        self.temporal_transformer = Transformer(t_dim, depth, heads, t_dim_head, t_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.weighted_mean = nn.Conv1d(in_channels=num_patches + 1, out_channels=1, kernel_size=1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(t_dim),
            nn.Linear(t_dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        x += self.spatial_pos_embedding[:, :n]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b t) n d -> b t (n d)', b=b, t=t)

        cls_tokens = repeat(self.temporal_cls_token, '() t d -> b t d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.temporal_pos_embedding[:, :(t + 1)]

        x = self.temporal_transformer(x)
        # x dim here is b, t, d
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = rearrange(x, '(b m) d -> b m d', m=2)
        x = x.mean(dim=1)
        return x


class Model3(nn.Module):
    def __init__(self, *, temporal_size, temporal_stride, num_joint, num_classes, dim,
                 depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0.5, emb_dropout=0.5):
        super().__init__()
        assert temporal_size % temporal_stride == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (temporal_size // temporal_stride)
        patch_dim = temporal_stride * num_joint * channels

        self.pre_bn = nn.BatchNorm1d(num_joint * channels)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('n c t v m -> (n m) t v c'),
            Rearrange('b (np ts) v c -> b np (ts v c)', np=num_patches, ts=temporal_stride),

            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim * num_joint),
            Rearrange('b np (v d) -> b np v d', v=num_joint)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, num_joint, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ST_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * num_joint),
            nn.Linear(dim * num_joint, num_classes)
        )

    def pre_process(self, x):
        n, c, t, v, m = x.shape
        x = rearrange(x, 'n c t v m -> (n m) (v c) t')
        x = self.pre_bn(x)
        x = rearrange(x, '(n m) (v c) t -> n c t v m', m=m, v=v)
        return x

    def forward(self, x):
        x = self.pre_process(x)
        x = self.to_patch_embedding(x)
        b, t, n, d = x.shape

        x += self.pos_embedding[:, :(t + 1), :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = rearrange(x, '(b m) t n d -> (b m) t (n d)', m=2)
        x = x.mean(dim=1)  # t dim

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = rearrange(x, '(b m) d -> b m d', m=2)
        x = x.mean(dim=1)
        return x


class Multi_scale_model(nn.Module):
    def __init__(self, *, temporal_size, temporal_stride_1, temporal_stride_2, num_joint, num_classes, dim,
                 depth, heads, mlp_dim, pool='cls', channels=3, partial_num=5,
                 dim_head=64, dropout=0.5, emb_dropout=0.5):
        super().__init__()
        assert temporal_size % temporal_stride_1 == 0, 'must be divisible by stride.'
        assert temporal_size % temporal_stride_2 == 0, 'must be divisible by stride.'
        num_tokens_1 = (temporal_size // temporal_stride_1)
        self.num_tokens_2 = (temporal_size // temporal_stride_2) * partial_num
        token_dim_1 = temporal_stride_1 * num_joint * channels
        self.partial_node = (num_joint // partial_num)
        self.token_dim_2 = temporal_stride_2 * self.partial_node * channels
        self.temporal_stride_2 = temporal_stride_2
        self.partial_num = partial_num
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.partial_point = torch.tensor([[10, 11, 12, 25, 24],
                                           [6, 7, 8, 23, 22],
                                           [9, 5, 2, 3, 4],
                                           [1, 17, 18, 19, 20],
                                           [1, 13, 14, 15, 16]]) - 1

        self.pre_bn = nn.BatchNorm1d(num_joint * channels)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('n c t v m -> (n m) t v c'),
            Rearrange('b (np ts) v c -> b np (ts v c)', np=num_tokens_1, ts=temporal_stride_1),
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(token_dim_1, dim),
        )

        self.patch_embedding_2_linear = nn.Linear(self.token_dim_2, dim)

        self.pos_embedding_1 = nn.Parameter(torch.randn(1, num_tokens_1 + 1, dim))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, self.num_tokens_2 + 1, dim))
        self.cls_token_1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_1 = nn.Dropout(emb_dropout)
        self.dropout_2 = nn.Dropout(emb_dropout)

        self.transformer_1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer_2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent_1 = nn.Identity()
        self.to_latent_2 = nn.Identity()

        self.mlp_head_1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
        self.mlp_head_2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def segment_embedding(self, x):
        n, c, t, v, m = x.shape
        x = rearrange(x, 'n c t v m->(n m) t v c')

        # 分割为5个部分
        partial = [x[:, :, self.partial_point[i], :]
                   for i in range(self.partial_num)]

        x = torch.cat(partial, dim=2)
        x = rearrange(x, 'b (t ts) (v pn) c -> b (t v) (ts pn c)',
                      ts=self.temporal_stride_2,
                      pn=self.partial_node)
        x = self.patch_embedding_2_linear(x)
        return x

    def pre_process(self, x):
        n, c, t, v, m = x.shape
        x = rearrange(x, 'n c t v m -> (n m) (v c) t')
        x = self.pre_bn(x)
        x = rearrange(x, '(n m) (v c) t -> n c t v m', m=m, v=v)
        return x

    def forward(self, x):
        x = self.pre_process(x)
        x_1 = self.to_patch_embedding(x)
        x_2 = self.segment_embedding(x)
        b_1, t_1, _ = x_1.shape
        b_2, t_2, _ = x_2.shape

        cls_tokens_1 = repeat(self.cls_token_1, '() n d -> b n d', b=b_1)
        cls_tokens_2 = repeat(self.cls_token_2, '() n d -> b n d', b=b_2)

        x_1 = torch.cat((cls_tokens_1, x_1), dim=1)
        x_1 += self.pos_embedding_1[:, :(t_1 + 1)]
        x_1 = self.dropout_1(x_1)

        x_1 = self.transformer_1(x_1)

        x_2 = torch.cat((cls_tokens_2, x_2), dim=1)
        x_2 += self.pos_embedding_2[:, :(t_2 + 1)]
        x_2 = self.dropout_2(x_2)

        x_2 = self.transformer_2(x_2)

        x_1 = rearrange(x_1, '(b m) n d -> b m n d', m=2)
        x_1 = x_1.mean(dim=1)  # mean two body frame
        x_1 = x_1.mean(dim=1) if self.pool == 'mean' else x_1[:, 0]
        x_1 = self.to_latent_1(x_1)
        x_1 = self.mlp_head_1(x_1)

        x_2 = rearrange(x_2, '(b m) n d -> b m n d', m=2)
        x_2 = x_2.mean(dim=1)  # mean two body frame
        x_2 = x_2.mean(dim=1) if self.pool == 'mean' else x_2[:, 0]
        x_2 = self.to_latent_2(x_2)
        x_2 = self.mlp_head_2(x_2)

        res = x_1 + x_2
        return res
