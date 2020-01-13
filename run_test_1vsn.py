#coding=utf-8
import test_1vsn_oa as tester

if __name__ == "__main__":
    argv = ['--devices', '2',
        '--test_name', '1000',
        
        '--image_height', '224',
        '--image_width', '224',
        '--num_dap_threads', '8',
        '--batch_size', '400',
        
        '--model_def', 'models.ShuffleNet_v2',
        #'--pretrained_model', 'train_models/20180909_221932_triplet_loss',
        '--pretrained_model', 'trained_models/san_train_20190705_223554',
        
        '--embedding_size', '128',
        '--use_batch_norm',
        '--use_normalized',
        '--fusion_method', 'single',
        '--top_num', '10',
        #'--save_feature'
        ]
    args = tester.parse_arguments(argv)
    tester.main(args)
