import argparse
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./dataset/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes in your custom dataset')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lamda0', type=float, default=1.0, help='coefficient for loss_arch')
parser.add_argument('--lamda1', type=float, default=1.0, help='coefficient for loss_aux')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
# parser.add_argument('--gpu_id', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=10, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='CELEB', help='experiment name')
parser.add_argument('--restart', type=str, default=None, help='restart directory')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--debug', action='store_true', default=False, help='debug')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--del_none', action='store_false', default=True, help='delete none operation')
parser.add_argument('--aux_loss', action='store_false', default=True, help='use info_entropy')
parser.add_argument('--beta_loss', action='store_false', default=True, help='use beta loss')
parser.add_argument('--info_linear_grow', action='store_true', default=False, help='info_linear_grow')
parser.add_argument('--one_level', action='store_true', default=False, help='use one level')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# parser.add_argument('--arch_name', type=str, default=None, required=True, help='save model name')
parser.add_argument('--loss_type', type=str, default='entropy', help='[entropy, mae, mse, rmse, kl]')
# 以下が2つ重要な新要素(最初のエポックでarch_paramを更新しないかどうか)
parser.add_argument('--warmup', action='store_true', default=False, help='warmup')
parser.add_argument('--warmup_epoch', type=int, default=15, help='total number of warmup epochs')
args = parser.parse_args()


image_size = (3,128,128)
train_rate = 0.8
validation_rate = 0.1
test_rate = 0.1
use_data_rate = 0.01
normalization = False
rotation_range=15.0
width_shift_range=0.15
height_shift_range=0.15
brightness_range = 0.0
shear_range=0.0
zoom_range=0.1
horizontal_flip=True
vertical_flip=False
# train_rate = None
# validation_rate = 0.018
# test_rate = 0.964
# use_data_rate = 1.0
# normalization = False
# rotation_range=0.0
# width_shift_range=0.0
# height_shift_range=0.0
# brightness_range = 0.0
# shear_range=0.0
# zoom_range=0.0
# horizontal_flip=False
# vertical_flip=False

args.image_size = image_size
args.train_rate = train_rate
args.validation_rate = validation_rate
args.test_rate = test_rate
args.use_data_rate = use_data_rate
args.normalization = normalization
args.rotation_range = rotation_range
args.width_shift_range = width_shift_range
args.height_shift_range = height_shift_range
args.brightness_range = brightness_range
args.shear_range = shear_range
args.zoom_range = zoom_range
args.horizontal_flip = horizontal_flip
args.vertical_flip = vertical_flip
