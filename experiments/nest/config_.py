
# Tempformer_model/pred_bias_matrix_vidvrd.npy
model_config = dict(
    num_enti_cats   = 36,
    num_pred_cats   = 133,
    dim_ffn         = 512,
    dim_enti        = 512,
    dim_pred        = 512,
    dim_att         = 512,
    dim_feat        = 1024,         # dimension of each bbox's RoI feature, depend on the detector
    dim_clsme       = 300,
    enco_pool_len   = 4,
    n_enco_layers   = 2,
    n_deco_layers   = 6,
    n_att_head      = 8,
    num_querys      = 192,
    neg_weight      = 0.1,
    positive_vIoU_th= 0.5,
    EntiNameEmb_path= "prepared_data/vidvrd_EntiNameEmb_pku.npy",
    bias_matrix_path= "prepared_data/pred_bias_matrix_vidvrd_pku.npy",
    cost_coeff_dict = dict(
        classification      = 1.0,
        adj_matrix          = 30.0,
    ), 
    loss_coeff_dict = dict(         # loss coefficient dictionary        
        classification      = 1.0,
        adj_matrix          = 30.0,
    )
)

test_dataset_config = dict(
    split = "test",
    ann_dir = "/nfshome/students/ik211072/VidSGG-BIG/datasets/nest",
    proposal_dir = "/nfshome/students/ik211072/VidSGG-BIG/tracking_results/nest",
    dim_boxfeature = 1024,
    min_frames_th = 5,
    max_proposal = 150,
    max_preds = 100,
    cache_tag = "nest"
)


inference_config = dict(
    topk = 10,
)
