import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import warnings

from detectron2.layers import Conv2d
from torch.nn.init import xavier_uniform_, constant_

from ....builder import HEADS
from .position_encoding import PositionEmbeddingSine3D

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1_box = nn.Dropout(dropout)
        self.norm1_box = nn.LayerNorm(d_model)

        # self attention for mask&class query
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

         # self attention for box query
        self.self_attn_box = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2_box = nn.Dropout(dropout)
        self.norm2_box = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)


        # ffn for box
        self.linear1_box = nn.Linear(d_model, d_ffn)
        self.activation_box = _get_activation_fn(activation)
        self.dropout3_box = nn.Dropout(dropout)
        self.linear2_box = nn.Linear(d_ffn, d_model)
        self.dropout4_box = nn.Dropout(dropout)
        self.norm3_box = nn.LayerNorm(d_model)

        self.time_attention_weights = nn.Linear(d_model, 1)
        self.time_attention = nn.Linear(d_model, d_model)

        self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        # normal_(self.level_embed)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_pos_embed_multf(tensor, pos):  # boardcase pos to every frame features
        return tensor if pos is None else tensor + pos.unsqueeze(1)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_ffn_box(self, tgt):
        tgt2 = self.linear2_box(self.dropout3_box(self.activation_box(self.linear1_box(tgt))))
        tgt = tgt + self.dropout4_box(tgt2)
        tgt = self.norm3_box(tgt)
        return tgt

    def forward(self, tgt, tgt_box, query_pos, src, pos, nf):
        '''
        tgt b q c == tgt_box b q c
        '''
        q1 = k1 = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q1.transpose(0, 1), k1.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt) # b q c

        assert len(tgt_box.shape) == 3  # tgt_box [bz,300,C]. # first layer

        # box-tgt self attention
        q_box = k_box = self.with_pos_embed(tgt_box, query_pos)
        tgt2_box = self.self_attn_box(q_box.transpose(0, 1), k_box.transpose(0, 1), tgt_box.transpose(0, 1))[0].transpose(0, 1)
        tgt_box = tgt_box + self.dropout2_box(tgt2_box)
        tgt_box = self.norm2_box(tgt_box) # b q c
        tgt2_box = self.cross_attn(
            query=self.with_pos_embed(tgt_box, query_pos).permute(1,0,2).unsqueeze(1).repeat(1,nf,1,1).flatten(1,2), # q tb c
            key=self.with_pos_embed(src, pos), # hw tb c 
            value=src)[0].permute(1,0,2) # hw tb c 
        _, qn, cn = tgt2_box.shape
        tgt2_box = tgt2_box.view(nf, -1, qn, cn).permute(1,0,2,3) # b t q c
        tgt_box = tgt_box.unsqueeze(1) + self.dropout1_box(tgt2_box)

        tgt_box = self.norm1_box(tgt_box)       
        # ffn box
        tgt_box = self.forward_ffn_box(tgt_box)

        return tgt, tgt_box

    def combine(self, tgt, tgt_box):
        '''
        tgt b q c
        tgt_box b t q c
        '''
        time_weight = self.time_attention_weights(tgt_box)
        time_weight = F.softmax(time_weight, 1)
        tgt2 = self.time_attention(tgt_box)
        tgt2 = (tgt2*time_weight).sum(1)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

@HEADS.register_module()
class VideoMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    # @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_frames = num_frames

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.deformabel_transformer_cross_attention_layers = DeformableTransformerDecoderLayer(
                                                                    d_model=hidden_dim, #256
                                                                    d_ffn=dim_feedforward, #1024
                                                                    dropout=0.1,
                                                                    n_heads=nheads, #8
                                                                    n_levels=1,
                                                                )

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm1 = nn.LayerNorm(hidden_dim)
        self.decoder_norm2 = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.class_embed_aux = nn.Linear(hidden_dim, num_classes + 1)

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.mask_embed_aux = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask = None):
        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        # disable mask, it does not affect performance
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            # self.pe_layer: not change dims, after flatten: bs, t, c, hw
            pos.append(self.pe_layer(x[i].view(bs, t, -1, size_list[-1][0], size_list[-1][1]), None).flatten(3))
            # self.input_proj: sequence(), after flatten: bt, c, hw
            # self.level_embed: embed for each layer: 1, c, 1
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            _, c, hw = src[-1].shape
            # NTxCxHW => NxTxCxHW => HWx(TN)xC
            pos[-1] = pos[-1].view(bs, t, c, hw).permute(3, 1, 0, 2).flatten(1,2)
            src[-1] = src[-1].view(bs, t, c, hw).permute(3, 1, 0, 2).flatten(1,2)

        # QxNxC (QxbsxC)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        
        query = query.permute(1,0,2) # b, q, c
        query_embed = query_embed.permute(1,0,2)

        predictions_class = []
        predictions_mask = []

        predictions_class_aux = []
        predictions_mask_aux = []

        src_flatten = torch.cat(src, 0)
        pos_flatten = torch.cat(pos, 0)
        query, query_frame = self.deformabel_transformer_cross_attention_layers(query, query, query_embed, src_flatten, pos_flatten, t)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            if i == 0:
                # prediction heads on learnable query features
                outputs_class, outputs_mask, attn_mask, outputs_class_aux, outputs_mask_aux =\
                                    self.forward_prediction_heads_all(query, 
                                    query_frame, mask_features, attn_mask_target_size=size_list[0])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_class_aux.append(outputs_class_aux)
                predictions_mask_aux.append(outputs_mask_aux)

            # attention: cross-attention first, query [bz, nf, 100, C] -> [100, nf*bz, C]
            bz, nf, q, c = query_frame.shape
            query_frame = query_frame.permute(2,1,0,3).flatten(1,2)
            query_pos = query_embed.unsqueeze(0).repeat(nf, 1, 1, 1).flatten(0,1).permute(1,0,2)
            query_frame = self.transformer_cross_attention_layers[i](
                query_frame, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_pos
            )

            query_frame = self.transformer_self_attention_layers[i](
                query_frame, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_pos
            )
            
            # FFN
            query_frame = self.transformer_ffn_layers[i](
                query_frame
            )

            # query  [100, nf*bz, C] -> [bz, nf, 100, C]
            query_frame = query_frame.view(q, nf, bz, c).permute(2,1,0,3)
            query = self.deformabel_transformer_cross_attention_layers.combine(query, query_frame)
            outputs_class, outputs_mask, attn_mask, outputs_class_aux, outputs_mask_aux =\
                                self.forward_prediction_heads_all(query, 
                                query_frame, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_class_aux.append(outputs_class_aux)
            predictions_mask_aux.append(outputs_mask_aux)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }

        out_aux = {
            'pred_logits': predictions_class_aux[-1],
            'pred_masks': predictions_mask_aux[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class_aux if self.mask_classification else None, predictions_mask_aux
            )
        }
        return out, out_aux

    def forward_output_frame(self, output_frame, mask_features):
        '''
        output_frame: b t q c
        '''
        decoder_output2 = self.decoder_norm2(output_frame)
        outputs_class = self.class_embed_aux(decoder_output2) # b t q c

        # mask_features bs, t, c_m, h_m, w_m
        mask_embed = self.mask_embed_aux(decoder_output2)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)
        # see output mask as 1f case
        return outputs_class.flatten(0,1), outputs_mask.flatten(0,1).unsqueeze(2)

    def forward_prediction_heads_all(self, output, output_frame, mask_features, attn_mask_target_size):
        '''
        output: b q c
        output_frame: b t q c
        '''
        decoder_output = self.decoder_norm1(output)
        outputs_class = self.class_embed(decoder_output)

        # mask_features bs, t, c_m, h_m, w_m
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
        b, q, t, _, _ = outputs_mask.shape

        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T, H*W] -> [B, h, Q, T, H*W] -> [T, B, h, Q, H*W] -> [T*B*h, Q, H*W]
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(3).unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1).permute(3, 0, 1, 2, 4).flatten(0, 2) < 0.5).bool()
        attn_mask = attn_mask.detach()
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        output_class_aux, output_mask_aux = self.forward_output_frame(output_frame, mask_features)

        return outputs_class, outputs_mask, attn_mask, output_class_aux, output_mask_aux


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
