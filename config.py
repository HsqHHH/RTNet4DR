import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--outf', default=' ', help='trained model will be saved at here')
    parser.add_argument('--save', default='UNet_test', help='save name of experiment in args.outf directory')
    parser.add_argument('--preprocess', type=str, default='7')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--crossentropy_weights5', default=[0.001, 0.1, 0.1, 1.0, 0.1])
    parser.add_argument('--crossentropy_weights2', default=[0.01, 1.])

    parser.add_argument('--train_patch_height', default=64)
    parser.add_argument('--train_patch_width', default=64)
    parser.add_argument('--N_patches', default=15, help='Number of training image patches')
    parser.add_argument('--inside_FOV', default='center', help='Choose from [not,center,all]')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--img_dir_train', type=str, default='')
    parser.add_argument('--img_dir_test', type=str, default='')
    parser.add_argument('--rotation', type=int, default=20)
    parser.add_argument('--sample_visualization', default=False, help='Visualization of training samples')
    parser.add_argument('--val_ratio', type=int, default=0.1)
    parser.add_argument('--val_on_test', default=True, type=bool)
    parser.add_argument('--train_data_path_list', default=' ')
    parser.add_argument('--test_data_path_list', default=' ')

    # model params
    parser.add_argument('--backbone', type=str, default='densenet161')
    parser.add_argument('--encoder_weights', type=str, default='imagenet')
    parser.add_argument('--lesion_classes', type=int, default=5)
    parser.add_argument('--vessel_classes', type=int, default=2)
    parser.add_argument('--activate', default=None, choices=[None, 'relu', 'softmax', 'sigmoid'])

    # save_dir params
    parser.add_argument('--dir_checkpoint', type=str, default=' ')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--model_subname', type=str, default='')

    args = parser.parse_args()
    return args