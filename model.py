from edcoder import PreModel
from adversarial_masking import AdversMask
from adversarial_masking_edge import AdversMaskEdge


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    encoder_type = args.encoder
    mask_encoder_type = args.mask_encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features
    drop_out = args.drop_out
    mlp = args.mlp
    num_proj_hidden = args.num_proj_hidden
    layer_dropout = args.layer_dropout
    fagcn_heads = args.fagcn_heads
    concat = args.concat
    add_edge_rate = args.add_edge_rate
    dropmessage = args.dropmessage

    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        drop_out=drop_out,
        mlp=mlp,
        num_proj_hidden=num_proj_hidden,
        layer_dropout=layer_dropout,
        fagcn_heads=fagcn_heads,
        concat=concat,
        add_edge_rate=add_edge_rate,
        dropmessage=dropmessage
    )

    mask_module = AdversMask(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        residual=residual,
        mask_encoder_type=mask_encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        max_degree=args.max_degree,
    )

    mask_module_edge = AdversMaskEdge(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        residual=residual,
        mask_encoder_type=mask_encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        max_degree=args.max_degree
    )

    return model, mask_module, mask_module_edge
