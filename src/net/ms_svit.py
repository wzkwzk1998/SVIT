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


class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads               # 有用的技巧
        if context is None:
            context = x

        if kv_include_self:
            context = torch.cat((x, context), dim=1)  # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)                # softmax

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


class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim,
                             PreNorm(lg_dim, Attention(dim=lg_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(lg_dim, sm_dim,
                             PreNorm(sm_dim, Attention(dim=sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]),
                                                                   (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens,
                                  kv_include_self=True) + sm_cls  # small class and large token
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens,
                                  kv_include_self=True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens


class Multi_scale_encoder(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 lg_depth,
                 sm_depth,
                 cross_depth,
                 dropout,
                 cross_dropout,
                 heads,
                 dim_head,
                 mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=dim, depth=lg_depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout),
                Transformer(dim=dim, depth=sm_depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout),
                CrossTransformer(sm_dim=dim, lg_dim=dim, depth=cross_depth, heads=heads, dim_head=dim_head,
                                 dropout=cross_dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens


class Multi_scale_model(nn.Module):
    """
    这个是cross的
    """

    def __init__(self, *, temporal_size, temporal_stride_1, temporal_stride_2, num_joint, num_classes, dim,
                 depth, lg_depth, sm_depth, cross_depth, heads, mlp_dim, pool='cls', channels=3, partial_num=5,
                 dim_head=64, dropout=0., emb_dropout=0., cross_dropout=0.):
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

        self.multi_scale_encoder = Multi_scale_encoder(dim=dim,
                                                       depth=depth,
                                                       lg_depth=lg_depth,
                                                       sm_depth=sm_depth,
                                                       cross_depth=cross_depth,
                                                       dropout=dropout,
                                                       cross_dropout=cross_dropout,
                                                       heads=heads,
                                                       dim_head=dim_head,
                                                       mlp_dim=mlp_dim)

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

        x_2 = torch.cat((cls_tokens_2, x_2), dim=1)
        x_2 += self.pos_embedding_2[:, :(t_2 + 1)]
        x_2 = self.dropout_2(x_2)

        x_1, x_2 = self.multi_scale_encoder(x_1, x_2)

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
