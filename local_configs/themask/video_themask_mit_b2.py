_base_ = [
    '../_base_/datasets/vspw_repeat2_mask2former.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw_mask2former.py'
]

valandtest_size = 1
data = dict(samples_per_gpu=2,
                train=dict(
                    dataset=dict(
                        dilation=[0])),
                val=dict(test_size=valandtest_size),
                test=dict(test_size=valandtest_size),
            )


evaluation = dict(interval=170000, metric='mIoU')

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
num_classes=124
num_object_queries=100
dec_layers=10
CONVS_DIM=256
MASK_DIM=256
nheads=8


model = dict(
    type='EncoderDecoder_clips',
    pretrained='/cluster/work/cvl/celiuce/video-seg/models/segformer/pretrained_models/mit_b2.pth',
    backbone=dict(
        type='mit_b2',
        style='pytorch'),

    decode_head=dict(
        type='THEMaskHead',
        num_classes=num_classes,
        loss_weight=1.0,
        ignore_value=255,
        input_shape=dict( #from mask2former backbone output_shape
            res2=dict(channels=64, stride=4,), #stride used in pixel decoder
            res3=dict(channels=128, stride=8,),
            res4=dict(channels=320, stride=16,),
            res5=dict(channels=512, stride=32,),
        ),
        in_index=[0, 1, 2, 3],
        in_features=["res2", "res3", "res4", "res5"], #named for fets used in pixel_decoder, is okay to be consistent with input_shape and transformer_in_features

        loss_cfg=dict(
            matcher_cfg=dict(
                class_weight=2.0,
                mask_weight=5.0,
                dice_weight=5.0,
                class_weight_unmatched=1.0,
                mask_weight_unmatched=0.5,
                dice_weight_unmatched=0.5,
            ),
            warmup_iters=30000,
            round_weight=[1.0, 0.5],
            aux_weight=0.25,
            deep_supervision=True,
            no_object_weight=0.1,
            train_num_points=12544,
            dec_layers=dec_layers,  # 9 decoder layers, add one for the loss on learnable query
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        ),
        pixel_decoder_cfg=dict(
            type='MSDeformAttnPixelDecoder',
            # params
            conv_dim=CONVS_DIM,
            mask_dim=MASK_DIM,
            norm="GN",
            transformer_dropout=0.0, #cfg.MODEL.MASK_FORMER.DROPOUT
            transformer_nheads=nheads, #cfg.MODEL.MASK_FORMER.NHEADS
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            transformer_in_features=["res3", "res4", "res5"], #cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
            common_stride=4,
        ),
        transformer_decoder_cfg=dict(
            type='VideoMultiScaleMaskedTransformerDecoder',
            CONVS_DIM=CONVS_DIM,
            MASK_DIM=MASK_DIM,
            transformer_in_feature="multi_scale_pixel_decoder",
            #params
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=num_object_queries, #cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            nheads=nheads,
            dim_feedforward=2048,
            dec_layers=dec_layers - 1,
            pre_norm=False,
            enforce_input_project=False,
            mask_dim=256,
            num_frames=1, 
        ),
        

    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
