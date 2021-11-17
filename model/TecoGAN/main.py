import sys
import tensorflow.contrib.slim as slim
import random as rn
from tensorflow.python.util import deprecation
import tensorflow as tf
import numpy as np
import os
import collections
import math
import time
import pdb
import keras
from scipy import signal
import cv2 as cv
from tensorflow.python.ops import summary_op_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False

# fix all randomness, except for multi-treading or GPU process
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)


Flags = tf.app.flags

Flags.DEFINE_integer('rand_seed', 1, 'random seed')
Flags.DEFINE_string('input_dir_LR', None,
                    'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_integer('input_dir_len', -1,
                     'length of the input for inference mode, -1 means all')
Flags.DEFINE_string('input_dir_HR', None,
                    'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_string('output_dir', None,
                    'The output directory of the checkpoint')
Flags.DEFINE_string(
    'output_pre', '', 'The name of the subfolder for the images')
Flags.DEFINE_string('output_name', 'output', 'The pre name of the outputs')
Flags.DEFINE_string('output_ext', 'png',
                    'The format of the output when evaluating')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('checkpoint', None,
                    'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_string('cudaID', '0', 'CUDA devices')
Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
#Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')

FLAGS = Flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cudaID
my_seed = FLAGS.rand_seed
rn.seed(my_seed)
np.random.seed(my_seed)
tf.set_random_seed(my_seed)

if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(FLAGS.summary_dir + "logfile.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


sys.stdout = Logger()

def fnet(fnet_input, reuse=False):
    def down_block( inputs, output_channel = 64, stride = 1, scope = 'down_block'):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
            net = lrelu(net, 0.2)
            net = conv2(net, 3, output_channel, stride, use_bias=True, scope='conv_2')
            net = lrelu(net, 0.2)
            net = maxpool(net)

        return net
        
    def up_block( inputs, output_channel = 64, stride = 1, scope = 'up_block'):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
            net = lrelu(net, 0.2)
            net = conv2(net, 3, output_channel, stride, use_bias=True, scope='conv_2')
            net = lrelu(net, 0.2)
            new_shape = tf.shape(net)[1:-1]*2
            net = tf.image.resize_images(net, new_shape)

        return net
        
    with tf.variable_scope('autoencode_unit', reuse=reuse):
        net = down_block( fnet_input, 32, scope = 'encoder_1')
        net = down_block( net, 64, scope = 'encoder_2')
        net = down_block( net, 128, scope = 'encoder_3')
        
        net = up_block( net, 256, scope = 'decoder_1')
        net = up_block( net, 128, scope = 'decoder_2')
        net1 = up_block( net, 64, scope = 'decoder_3')
        
        with tf.variable_scope('output_stage'):
            net = conv2(net1, 3, 32, 1, scope='conv1')
            net = lrelu(net, 0.2)
            net2 = conv2(net, 3, 2, 1, scope='conv2')
            net = tf.tanh(net2) * 24.0
            # the 24.0 is the max Velocity, details can be found in TecoGAN paper
    return net

# Definition of the generator, more details can be found in TecoGAN paper
def generator_F(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise  ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channel = 64, stride = 1, scope = 'res_block'):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
            net = tf.nn.relu(net)
            net = conv2(net, 3, output_channel, stride, use_bias=True, scope='conv_2')
            net = net + inputs

        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 3, 64, 1, scope='conv')
            stage1_output = tf.nn.relu(net)

        net = stage1_output

        # The residual block parts
        for i in range(1, FLAGS.num_resblock+1 , 1): # should be 16 for TecoGAN, and 10 for TecoGANmini
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('conv_tran2highres'):
            net = conv2_tran(net, 3, 64, 2, scope='conv_tran1')
            net = tf.nn.relu(net)
            
            net = conv2_tran(net, 3, 64, 2, scope='conv_tran2')
            net = tf.nn.relu(net)
        
        with tf.variable_scope('output_stage'):
            net = conv2(net, 3, gen_output_channels, 1, scope='conv')
            low_res_in = gen_inputs[:,:,:,0:3] # ignore warped pre high res
            # for tensoflow API<=1.13, bicubic_four is equivalent to the followings:
            # hi_shape = tf.shape( net )
            # bicubic_hi = tf.image.resize_bicubic( low_res_in, (hi_shape[1], hi_shape[2])) # no GPU implementation
            bicubic_hi = bicubic_four( low_res_in ) # can put on GPU
            net = net + bicubic_hi
            net = preprocess( net )
    return net

def inference_data_loader(FLAGS):

    filedir = FLAGS.input_dir_LR
    downSP = False
    if (FLAGS.input_dir_LR is None) or (not os.path.exists(FLAGS.input_dir_LR)):
        if (FLAGS.input_dir_HR is None) or (not os.path.exists(FLAGS.input_dir_HR)):
            raise ValueError('Input directory not found')
        filedir = FLAGS.input_dir_HR
        downSP = True
        
    image_list_LR_temp = os.listdir(filedir)
    image_list_LR_temp = [_ for _ in image_list_LR_temp if _.endswith(".png")] 
    image_list_LR_temp = sorted(image_list_LR_temp) # first sort according to abc, then sort according to 123
    image_list_LR_temp.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
    if FLAGS.input_dir_len > 0:
        image_list_LR_temp = image_list_LR_temp[:FLAGS.input_dir_len]
        
    image_list_LR = [os.path.join(filedir, _) for _ in image_list_LR_temp]

    # Read in and preprocess the images
    def preprocess_test(name):
        im = cv.imread(name,3).astype(np.float32)[:,:,::-1]
        
        if downSP:
            icol_blur = cv.GaussianBlur( im, (0,0), sigmaX = 1.5)
            im = icol_blur[::4,::4,::]
        im = im / 255.0 #np.max(im)
        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]
    
    if True: # a hard-coded symmetric padding
        image_list_LR = image_list_LR[5:0:-1] + image_list_LR
        image_LR = image_LR[5:0:-1] + image_LR

    Data = collections.namedtuple('Data', 'paths_LR, inputs')
    print(image_LR)
    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )

# load hi-res data from disk
def loadHR_batch(FLAGS, tar_size):
    # file processing on CPU
    with tf.device('/cpu:0'):
        #Check the input directory
        if (FLAGS.input_video_dir == ''):
            raise ValueError('Video input directory input_video_dir is not provided')
        if (not os.path.exists(FLAGS.input_video_dir)):
            raise ValueError('Video input directory not found')
            
        image_set_lists = []
        with tf.variable_scope('load_frame'):
            for dir_i in range(FLAGS.str_dir, FLAGS.end_dir+1):
                inputDir = os.path.join( FLAGS.input_video_dir, '%s_%04d' %(FLAGS.input_video_pre, dir_i) )
                if (os.path.exists(inputDir)): # the following names are hard coded: col_high_
                    if not os.path.exists(os.path.join(inputDir ,'col_high_%04d.png' % FLAGS.max_frm) ):
                        print("Skip %s, since foler doesn't contain enough frames!" % inputDir)
                        continue
                        
                    image_list = [ os.path.join(inputDir ,'col_high_%04d.png' % frame_i ) 
                                    for frame_i in range(FLAGS.max_frm + 1 )]
                    image_set_lists.append(image_list) 
            tensor_set_lists = tf.convert_to_tensor(image_set_lists, dtype=tf.string)
            input_slices = tf.train.slice_input_producer([tensor_set_lists], shuffle=False,
                                            capacity=int(FLAGS.name_video_queue_capacity) )
            
            input_slices = input_slices[0] # one slice contains pathes for (FLAGS.max_frm frames + 1) frames
            
            HR_data = []
            for frame_i in range(FLAGS.max_frm + 1 ):
                HR_data_i =  tf.image.convert_image_dtype( 
                                    tf.image.decode_png(tf.read_file(input_slices[frame_i]), channels=3), 
                                    dtype=tf.float32)
                HR_data += [HR_data_i]
            # the reshape after loading is necessary as the unkown output shape crashes the cropping op
            HR_data = tf.stack(HR_data, axis=0)
            HR_shape = tf.shape(HR_data)#(FLAGS.max_frm+1, h, w, 3))
            HR_data = tf.reshape(HR_data, (HR_shape[0], HR_shape[1], HR_shape[2], HR_shape[3]))
            
        # sequence preparation and data augmentation part
        with tf.name_scope('sequence_data_preprocessing'):
            sequence_length = FLAGS.max_frm - FLAGS.RNN_N + 1
            num_image_list_HR_t_cur = len(image_set_lists)*sequence_length
            HR_sequences, name_sequences = [], []
            if FLAGS.random_crop and FLAGS.mode == 'train':
                print('[Config] Use random crop')
                # a k_w_border margin is in tar_size for gaussian blur
                # have to use the same crop because crop_to_bounding_box only accept one value
                offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, \
                    tf.cast(HR_shape[-2], tf.float32) - tar_size )),dtype=tf.int32)
                offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, \
                    tf.cast(HR_shape[-3], tf.float32) - tar_size )),dtype=tf.int32) 
            else:
                raise Exception('Not implemented') # train_data can have different resolutions, crop is necessary    
            
            if(FLAGS.movingFirstFrame): 
                print('[Config] Move first frame')
                # our data augmentation, moving first frame to mimic camera motion
                # random motions, one slice use the same motion
                offset_xy = tf.cast(tf.floor(tf.random_uniform([FLAGS.RNN_N, 2], -3.5,4.5)),dtype=tf.int32)
                # [FLAGS.RNN_N , 2], relative positions
                pos_xy = tf.cumsum(offset_xy, axis=0, exclusive=True) 
                # valid regions, lefttop_pos, target_size-range_pos
                min_pos = tf.reduce_min( pos_xy, axis=0 )
                range_pos = tf.reduce_max( pos_xy, axis=0 ) - min_pos # [ shrink x, shrink y ]
                lefttop_pos = pos_xy - min_pos # crop point
                moving_decision = tf.random_uniform([sequence_length], 0, 1, dtype=tf.float32)
                fix_off_h = tf.clip_by_value(offset_h, 0, HR_shape[-3] - tar_size - range_pos[1])
                fix_off_w = tf.clip_by_value(offset_w, 0, HR_shape[-2] - tar_size - range_pos[0])
            
            if FLAGS.flip and (FLAGS.mode == 'train'):
                print('[Config] Use random flip')
                # Produce the decision of random flip
                flip_decision = tf.random_uniform([sequence_length], 0, 1, dtype=tf.float32)
                
            for fi in range( FLAGS.RNN_N ):
                HR_sequence = HR_data[ fi : fi+sequence_length ]  # sequence_length, h, w, 3
                name_sequence = input_slices[ fi : fi+sequence_length ]
                
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    HR_sequence = random_flip_batch(HR_sequence, flip_decision)
                    
                # currently, it is necessary to crop, because training videos have different resolutions
                if FLAGS.random_crop and FLAGS.mode == 'train':
                    HR_sequence_crop = tf.image.crop_to_bounding_box(HR_sequence, 
                            offset_h, offset_w, tar_size, tar_size)
                            
                    if FLAGS.movingFirstFrame:
                        if(fi == 0): # always use the first frame
                            HR_data_0 = tf.identity(HR_sequence)
                            name_0 = tf.identity(name_sequence)
                            
                        name_sequence1 = name_0
                        HR_sequence_crop_1 = tf.image.crop_to_bounding_box(HR_data_0, 
                            fix_off_h + lefttop_pos[fi][1], fix_off_w + lefttop_pos[fi][0], tar_size, tar_size)
                            
                        HR_sequence_crop = tf.where(moving_decision < 0.7, HR_sequence_crop, HR_sequence_crop_1)
                        name_sequence = tf.where(moving_decision < 0.7, name_sequence, name_sequence1) 
                

                HR_sequence_crop.set_shape([sequence_length, tar_size, tar_size, 3])
                HR_sequences.append( HR_sequence_crop )
                name_sequences.append( name_sequence )
                
            target_images = HR_sequences # RNN_N, sequence_length,tar_size,tar_size,3
            output_names = name_sequences
            
        if len(target_images)!=FLAGS.RNN_N:
            raise ValueError('Length of target image sequence is incorrect,expected {}, got {}.'.format(FLAGS.RNN_N, len(target_images)))
        
        print('Sequenced batches: {}, sequence length: {}'.format(num_image_list_HR_t_cur, FLAGS.RNN_N))
        batch_list = tf.train.shuffle_batch(output_names + target_images, enqueue_many=True,\
                        batch_size=int(FLAGS.batch_size), capacity=FLAGS.video_queue_capacity+FLAGS.video_queue_batch*sequence_length,
                        min_after_dequeue=FLAGS.video_queue_capacity, num_threads=FLAGS.queue_thread, seed = FLAGS.rand_seed)
        
    return batch_list, num_image_list_HR_t_cur # a k_w_border margin is in there for gaussian blur
    
# load hi-res data from disk
def loadHR(FLAGS, tar_size):
    # a k_w_border margin should be in tar_size for Gaussian blur
    
    with tf.device('/cpu:0'):
        #Check the input directory
        if (FLAGS.input_video_dir == ''):
            raise ValueError('Video input directory input_video_dir is not provided')

        if (not os.path.exists(FLAGS.input_video_dir)):
            raise ValueError('Video input directory not found')

        image_list_HR_r = [[] for _ in range( FLAGS.RNN_N )] # all empty lists
        
        for dir_i in range(FLAGS.str_dir, FLAGS.end_dir+1):
            inputDir = os.path.join( FLAGS.input_video_dir, '%s_%04d' %(FLAGS.input_video_pre, dir_i) )
            if (os.path.exists(inputDir)): # the following names are hard coded
                if not os.path.exists(os.path.join(inputDir ,'col_high_%04d.png' % FLAGS.max_frm) ):
                    print("Skip %s, since foler doesn't contain enough frames!" % inputDir)
                    continue
                for fi in range( FLAGS.RNN_N ):
                    image_list_HR_r[fi] += [ os.path.join(inputDir ,'col_high_%04d.png' % frame_i ) 
                                        for frame_i in range(fi, FLAGS.max_frm - FLAGS.RNN_N + fi + 1 )]
        
        num_image_list_HR_t_cur = len(image_list_HR_r[0])
        if num_image_list_HR_t_cur==0:
            raise Exception('No frame files in the video input directory')
                    
        image_list_HR_r = [tf.convert_to_tensor(_ , dtype=tf.string) for _ in image_list_HR_r ]

        with tf.variable_scope('load_frame'):
            # define the image list queue
            output = tf.train.slice_input_producer(image_list_HR_r, shuffle=False,\
                capacity=int(FLAGS.name_video_queue_capacity) )
            output_names = output
            
            data_list_HR_r = []# high res rgb, in range 0-1, shape any
                
            if FLAGS.movingFirstFrame and FLAGS.mode == 'train': # our data augmentation, moving first frame to mimic camera motion
                print('[Config] Use random crop')
                offset_xy = tf.cast(tf.floor(tf.random_uniform([FLAGS.RNN_N, 2], -3.5,4.5)),dtype=tf.int32)
                # [FLAGS.RNN_N , 2], shifts
                pos_xy = tf.cumsum(offset_xy, axis=0, exclusive=True) # relative positions
                min_pos = tf.reduce_min( pos_xy, axis=0 )
                range_pos = tf.reduce_max( pos_xy, axis=0 ) - min_pos # [ shrink x, shrink y ]
                lefttop_pos = pos_xy - min_pos # crop point
                moving_decision = tf.random_uniform([], 0, 1, dtype=tf.float32)
                
            for fi in range( FLAGS.RNN_N ):
                HR_data = tf.image.convert_image_dtype( tf.image.decode_png(tf.read_file(output[fi]), channels=3), dtype=tf.float32)
                if(FLAGS.movingFirstFrame):
                    if(fi == 0):
                        HR_data_0 = tf.identity(HR_data)
                        target_size = tf.shape(HR_data_0)
                        
                    HR_data_1 = tf.image.crop_to_bounding_box(HR_data_0, 
                        lefttop_pos[fi][1], lefttop_pos[fi][0], 
                        target_size[0] - range_pos[1], target_size[1] - range_pos[0])
                    HR_data = tf.cond(moving_decision < 0.7, lambda: tf.identity(HR_data), lambda: tf.identity(HR_data_1))
                    output_names[fi] = tf.cond(moving_decision < 0.7, lambda: tf.identity(output_names[fi]), lambda: tf.identity(output_names[0]))
                data_list_HR_r.append( HR_data )
            
            target_images = data_list_HR_r
        
        # Other data augmentation part
        with tf.name_scope('data_preprocessing'):
            
            with tf.name_scope('random_crop'):
                # Check whether perform crop
                if (FLAGS.random_crop is True) and FLAGS.mode == 'train':
                    print('[Config] Use random crop')
                    target_size = tf.shape(target_images[0])
                    
                    offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, \
                        tf.cast(target_size[1], tf.float32) - tar_size )),dtype=tf.int32)
                    offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, \
                        tf.cast(target_size[0], tf.float32) - tar_size )),dtype=tf.int32) 
                    
                    for frame_t in range(FLAGS.RNN_N):
                        target_images[frame_t] = tf.image.crop_to_bounding_box(target_images[frame_t], 
                            offset_h, offset_w, tar_size, tar_size) 
                        
                else:
                    raise Exception('Not implemented')
            
            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    print('[Config] Use random flip')
                    # Produce the decision of random flip
                    flip_decision = tf.random_uniform([], 0, 1, dtype=tf.float32)
                    for frame_t in range(FLAGS.RNN_N):
                        target_images[frame_t] = random_flip(target_images[frame_t], flip_decision)
                    
            for frame_t in range(FLAGS.RNN_N):
                target_images[frame_t].set_shape([tar_size, tar_size, 3])
                
        if FLAGS.mode == 'train':
            print('Sequenced batches: {}, sequence length: {}'.format(num_image_list_HR_t_cur, FLAGS.RNN_N))
            batch_list = tf.train.shuffle_batch(output_names + target_images,\
                            batch_size=int(FLAGS.batch_size), capacity=FLAGS.video_queue_capacity+FLAGS.video_queue_batch*FLAGS.max_frm,
                            min_after_dequeue=FLAGS.video_queue_capacity, num_threads=FLAGS.queue_thread, seed = FLAGS.rand_seed)
        else:
            raise Exception('Not implemented')
    return batch_list, num_image_list_HR_t_cur # a k_w_border margin is still there for gaussian blur!!


def frvsr_gpu_data_loader(FLAGS, useValData_ph): # useValData_ph, tf bool placeholder, whether to use validationdata
    Data = collections.namedtuple('Data', 'paths_HR, s_inputs, s_targets, image_count, steps_per_epoch')
    tar_size = FLAGS.crop_size
    tar_size = (FLAGS.crop_size * 4 ) + int(1.5 * 3.0) * 2 # crop_size * 4, and Gaussian blur margin
    k_w_border = int(1.5 * 3.0)
    
    loadHRfunc = loadHR if FLAGS.queue_thread > 4 else loadHR_batch
    # loadHR_batch load 120 frames at once, is faster for a single queue, and usually will be slower and slower for larger queue_thread
    # loadHR load RNN_N frames at once, is faster when queue_thread > 4, but slow for queue_thread < 4
    
    with tf.name_scope('load_frame_cpu'):
        with tf.name_scope('train_data'):
            print("Preparing train_data")
            batch_list, num_image_list_HR_t_cur = loadHRfunc(FLAGS, tar_size)
        with tf.name_scope('validation_data'):
            print("Preparing validation_data")
            val_capacity = 128 # TODO parameter!
            val_q_thread = 1   # TODO parameter!
            valFLAGS = copy_update_configuration(FLAGS, \
                {"str_dir":FLAGS.end_dir + 1,"end_dir":FLAGS.end_dir_val,"name_video_queue_capacity":val_capacity,\
                    "video_queue_capacity":val_capacity, "queue_thread":val_q_thread})
            vald_batch_list, vald_num_image_list_HR_t_cur = loadHRfunc(valFLAGS, tar_size)
            
    HR_images = list(batch_list[FLAGS.RNN_N::])# batch high-res images
    HR_images_vald = list(vald_batch_list[FLAGS.RNN_N::])# test batch high-res images
    
    steps_per_epoch = num_image_list_HR_t_cur // FLAGS.batch_size
    
    target_images = []
    input_images = []
    with tf.name_scope('load_frame_gpu'):
        with tf.device('/gpu:0'):
            for frame_t in range(FLAGS.RNN_N):
                def getTrainHR():
                    return HR_images[frame_t]
                def getValdHR():
                    return HR_images_vald[frame_t]
                    
                curHR = tf.cond( useValData_ph, getValdHR, getTrainHR )
                input_images.append( tf_data_gaussDownby4(curHR, 1.5) )
                
                input_images[frame_t].set_shape([FLAGS.batch_size,FLAGS.crop_size, FLAGS.crop_size, 3])
                input_images[frame_t] = preprocessLR(input_images[frame_t])
                
                target_images.append(tf.image.crop_to_bounding_box(curHR, 
                                k_w_border, k_w_border, \
                                FLAGS.crop_size*4,\
                                FLAGS.crop_size*4) )
                target_images[frame_t] = preprocess(target_images[frame_t])
                target_images[frame_t].set_shape([FLAGS.batch_size,FLAGS.crop_size*4, FLAGS.crop_size*4, 3])
            
            
            # for Ds, inputs_batch and targets_batch are just the input and output:
            S_inputs_frames = tf.stack(input_images, axis = 1) # batch, frame, FLAGS.crop_size, FLAGS.crop_size, sn
            S_targets_frames = tf.stack(target_images, axis = 1) # batch, frame, FLAGS.crop_size*4, FLAGS.crop_size*4, 3
            S_inputs_frames.set_shape( (FLAGS.batch_size,FLAGS.RNN_N,FLAGS.crop_size,FLAGS.crop_size,3) )
            S_targets_frames.set_shape( (FLAGS.batch_size,FLAGS.RNN_N,4*FLAGS.crop_size,4*FLAGS.crop_size,3) )
        
    #Data = collections.namedtuple('Data', 'paths_HR, s_inputs, s_targets, image_count, steps_per_epoch')
    def getTrainHRpath():
        return batch_list[:FLAGS.RNN_N]
    def getValdHRpath():
        return vald_batch_list[:FLAGS.RNN_N]
        
    curHRpath = tf.cond( useValData_ph, getValdHRpath, getTrainHRpath )
                
    return Data(
        paths_HR=curHRpath,
        s_inputs=S_inputs_frames,       # batch, frame, FLAGS.crop_size, FLAGS.crop_size, sn
        s_targets=S_targets_frames,     # batch, frame, FLAGS.crop_size*4, FLAGS.crop_size*4, 3
        image_count=num_image_list_HR_t_cur,
        steps_per_epoch=steps_per_epoch
    )


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocessLR(image):
    with tf.name_scope("preprocessLR"):
        return tf.identity(image)


def deprocessLR(image):
    with tf.name_scope("deprocessLR"):
        return tf.identity(image)

# Define the convolution transpose building block
def conv2_tran(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)

# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)


def conv2_NCHW(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv_NCHW'):
    # Use NCWH to speed up the inference
    # kernel: list of 2 integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NCWH',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NCWH',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                               biases_initializer=None)


# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), \
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.MODEL_VARIABLES ],dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


# Define our Lrelu
def lrelu(inputs, alpha):
    return keras.layers.LeakyReLU(alpha=alpha).call(inputs)


def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)

def maxpool(inputs, scope='maxpool'):
    return slim.max_pool2d(inputs, [2, 2], scope=scope)
    
# Our dense layer
def denselayer(inputs, output_size):
    # Rachel todo, put it to Model variable_scope
    denseLayer = tf.layers.Dense(output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    output = denseLayer.apply(inputs)
    tf.add_to_collection( name=tf.GraphKeys.MODEL_VARIABLES, value=denseLayer.kernel )
    #output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    return output

# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output
    
def upscale_four(inputs, scope='upscale_four'): # mimic the tensorflow bilinear-upscaling for a fix ratio of 4
    with tf.variable_scope(scope):
        size = tf.shape(inputs)
        b = size[0]
        h = size[1]
        w = size[2]
        c = size[3]
        
        p_inputs = tf.concat( (inputs, inputs[:,-1:,:,:] ), axis = 1) # pad bottom
        p_inputs = tf.concat( (p_inputs, p_inputs[:,:,-1:,:] ), axis = 2) # pad right
        
        hi_res_bin = [ 
            [
                inputs, # top-left
                p_inputs[:,:-1,1:,:] # top-right
            ], 
            [
                p_inputs[:,1:,:-1,:], # bottom-left
                p_inputs[:,1:,1:,:] # bottom-right
            ]
        ]
        
        hi_res_array = [] 
        for hi in range(4):
            for wj in range(4):
                hi_res_array.append( 
                    hi_res_bin[0][0] * (1.0 - 0.25 * hi) * (1.0 - 0.25 * wj) 
                    + hi_res_bin[0][1] * (1.0 - 0.25 * hi) * (0.25 * wj) 
                    + hi_res_bin[1][0] * (0.25 * hi) * (1.0 - 0.25 * wj) 
                    + hi_res_bin[1][1] * (0.25 * hi) * (0.25 * wj) 
                    )
                    
        hi_res =  tf.stack( hi_res_array, axis = 3 ) # shape (b,h,w,16,c)
        hi_res_reshape = tf.reshape( hi_res, (b, h, w, 4, 4, c) )
        hi_res_reshape = tf.transpose( hi_res_reshape, (0,1,3,2,4,5) )
        hi_res_reshape = tf.reshape( hi_res_reshape, (b, h*4, w*4, c) )
    
    return hi_res_reshape
    
    
def bicubic_four(inputs, scope='bicubic_four'): 
    '''
        equivalent to tf.image.resize_bicubic( inputs, (h*4, w*4) ) for a fix ratio of 4 FOR API <=1.13
        For API 2.0, tf.image.resize_bicubic will be different, old version is tf.compat.v1.image.resize_bicubic
        **Parallel Catmull-Rom Spline Interpolation Algorithm for Image Zooming Based on CUDA*[Wu et. al.]**
    '''
    
    with tf.variable_scope(scope):
        size = tf.shape(inputs)
        b = size[0]
        h = size[1]
        w = size[2]
        c = size[3]
        
        p_inputs = tf.concat( (inputs[:,:1,:,:],   inputs)  , axis = 1) # pad top 
        p_inputs = tf.concat( (p_inputs[:,:,:1,:], p_inputs), axis = 2) # pad left
        p_inputs = tf.concat( (p_inputs, p_inputs[:,-1:,:,:], p_inputs[:,-1:,:,:]), axis = 1) # pad bottom
        p_inputs = tf.concat( (p_inputs, p_inputs[:,:,-1:,:], p_inputs[:,:,-1:,:]), axis = 2) # pad right
        
        hi_res_bin = [p_inputs[:,bi:bi+h,:,:] for bi in range(4) ]
        r = 0.75
        mat = np.float32( [[0,1,0,0],[-r,0,r,0], [2*r,r-3,3-2*r,-r], [-r,2-r,r-2,r]] )
        weights = [np.float32([1.0, t, t*t, t*t*t]).dot(mat) for t in [0.0, 0.25, 0.5, 0.75]]
        
        hi_res_array = [] # [hi_res_bin[1]] 
        for hi in range(4):
            cur_wei = weights[hi]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
                
            hi_res_array.append(cur_data)
            
        hi_res_y =  tf.stack( hi_res_array, axis = 2 ) # shape (b,h,4,w,c)
        hi_res_y = tf.reshape( hi_res_y, (b, h*4, w+3, c) )
        
        hi_res_bin = [hi_res_y[:,:,bj:bj+w,:] for bj in range(4) ]
        
        hi_res_array = [] # [hi_res_bin[1]]
        for hj in range(4):
            cur_wei = weights[hj]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
                
            hi_res_array.append(cur_data)
            
        hi_res =  tf.stack( hi_res_array, axis = 3 ) # shape (b,h*4,w,4,c)
        hi_res = tf.reshape( hi_res, (b, h*4, w*4, c) )
        
    return hi_res

def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

# The random flip operation used for loading examples of one batch
def random_flip_batch(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.where(tf.less(decision, 0.5), f2, f1)

    return output

# The random flip operation used for loading examples
def random_flip(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)

    return output

# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    for name, value in FLAGS.flag_values_dict().items():
        print('\t%s: %s'%(name, str(value)))
    print('End of configuration')
    

def copy_update_configuration(FLAGS, updateDict = {}):
    namelist = []
    valuelist = []
    for name, value in FLAGS.flag_values_dict().items():
        namelist += [name] 
        if( name in updateDict):
            valuelist += [updateDict[name]]
        else:
            valuelist += [value]
    Params = collections.namedtuple('Params', ",".join(namelist))
    tmpFLAGS = Params._make(valuelist)
    #print(tmpFLAGS)
    return tmpFLAGS
    
def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))

    return psnr

# VGG19 component
def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
        return arg_sc

# VGG19 net
def vgg_19(inputs,
           num_classes=1000,        # no effect
           is_training=False,       # no effect
           dropout_keep_prob=0.5,   # no effect
           spatial_squeeze=True,    # no effect
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
  """Changed from the Oxford Net VGG 19-Layers version E Example.
  Note: Only offer features from conv1 until relu54, classification part is removed
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # fully_connected layers are skipped here! because we only need the feature maps
      #     from the previous layers
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points
# vgg_19.default_image_size = 224


### Helper functions for data loading ############################################################
def gaussian_2dkernel(size=5, sig=1.):
    """
    Returns a 2D Gaussian kernel array with side length size and a sigma of sig
    """
    gkern1d = signal.gaussian(size, std=sig).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return (gkern2d/gkern2d.sum())
    
def tf_data_gaussDownby4( HRdata, sigma = 1.5 ):
    """
    tensorflow version of the 2D down-scaling by 4 with Gaussian blur
    sigma: the sigma used for Gaussian blur
    return: down-scaled data
    """
    k_w = 1 + 2 * int(sigma * 3.0)
    gau_k = gaussian_2dkernel(k_w, sigma)
    gau_0 = np.zeros_like(gau_k)
    gau_list = np.float32(  [
        [gau_k, gau_0, gau_0],
        [gau_0, gau_k, gau_0],
        [gau_0, gau_0, gau_k]]  ) # only works for RGB images!
    gau_wei = np.transpose( gau_list, [2,3,0,1] )
    
    with tf.device('/gpu:0'):
        fix_gkern = tf.constant( gau_wei, dtype = tf.float32, shape = [k_w, k_w, 3, 3], name='gauss_blurWeights' )
        # shape [batch_size, crop_h, crop_w, 3]
        cur_data = tf.nn.conv2d(HRdata, fix_gkern, strides=[1,4,4,1], padding="VALID", name='gauss_downsample_4')
    
        return cur_data
        
### Helper functions for model loading ############################################################        
def get_existing_from_ckpt(ckpt, var_list=None, rest_zero=False, print_level=1):
    reader = tf.train.load_checkpoint(ckpt)
    ops = []
    if(var_list is None):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in var_list:
        tensor_name = var.name.split(':')[0]
        if reader.has_tensor(tensor_name):
            npvariable = reader.get_tensor(tensor_name)
            if(print_level >= 2):
                print ("loading tensor: " + str(var.name) + ", shape " + str(npvariable.shape))
            if( var.shape != npvariable.shape ):
                raise ValueError('Wrong shape in for {} in ckpt,expected {}, got {}.'.format(var.name, str(var.shape),
                    str(npvariable.shape)))
            ops.append(var.assign(npvariable))
        else:
            if(print_level >= 1): print("variable not found in ckpt: " + var.name)
            if rest_zero:
                if(print_level >= 1): print("Assign Zero of " + str(var.shape))
                npzeros = np.zeros((var.shape))
                ops.append(var.assign(npzeros))
    return ops
    
# gif summary
"""gif_summary_v2.ipynb, Original file is located at
[a future version] https://colab.research.google.com/drive/1CSOrCK8-iQCZfs3CVchLE42C52M_3Sej
[current version]  https://colab.research.google.com/drive/1vgD2HML7Cea_z5c3kPBcsHUIxaEVDiIc
"""

def encode_gif(images, fps):
    """Encodes numpy images into gif string.
    Args:
      images: A 5-D `uint8` `np.array` (or a list of 4-D images) of shape
        `[batch_size, time, height, width, channels]` where `channels` is 1 or 3.
      fps: frames per second of the animation
    Returns:
      The encoded gif string.
    Raises:
      IOError: If the ffmpeg command returns an error.
    """
    from subprocess import Popen, PIPE
    h, w, c = images[0].shape
    cmd = ['ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-r', '%.02f' % fps,
        '-s', '%dx%d' % (w, h),
        '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
        '-file', '-',
        '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
        '-r', '%.02f' % fps,
        '-f', 'gif',
        '-']
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc
    return out


def py_gif_summary(tag, images, max_outputs, fps):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
      tag: Name of the summary.
      images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
        channels]` where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
    Returns:
      The serialized `Summary` protocol buffer.
    Raises:
      ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
    """
    is_bytes = isinstance(tag, bytes)
    if is_bytes:
        tag = tag.decode("utf-8")
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise ValueError("Tensor must have dtype uint8 for gif summary.")
    if images.ndim != 5:
        raise ValueError("Tensor must be 5-D for gif summary.")
    batch_size, _, height, width, channels = images.shape
    if channels not in (1, 3):
        raise ValueError("Tensors must have 1 or 3 channels for gif summary.")
    
    summ = tf.Summary()
    num_outputs = min(batch_size, max_outputs)
    for i in range(num_outputs):
        image_summ = tf.Summary.Image()
        image_summ.height = height
        image_summ.width = width
        image_summ.colorspace = channels  # 1: grayscale, 3: RGB
        try:
            image_summ.encoded_image_string = encode_gif(images[i], fps)
        except (IOError, OSError) as e:
            tf.logging.warning("Unable to encode images to a gif string because either ffmpeg is "
                "not installed or ffmpeg returned an error: %s. Falling back to an "
                "image summary of the first frame in the sequence.", e)
            try:
                from PIL import Image  # pylint: disable=g-import-not-at-top
                import io  # pylint: disable=g-import-not-at-top
                with io.BytesIO() as output:
                    Image.fromarray(images[i][0]).save(output, "PNG")
                    image_summ.encoded_image_string = output.getvalue()
            except:
                tf.logging.warning("Gif summaries requires ffmpeg or PIL to be installed: %s", e)
                image_summ.encoded_image_string = "".encode('utf-8') if is_bytes else ""
        if num_outputs == 1:
            summ_tag = "{}/gif".format(tag)
        else:
            summ_tag = "{}/gif/{}".format(tag, i)
        summ.value.add(tag=summ_tag, image=image_summ)
    summ_str = summ.SerializeToString()
    return summ_str

def gif_summary(name, tensor, max_outputs, fps, collections=None, family=None):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
      name: Name of the summary.
      tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
        channels]` where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
      collections: Optional list of tf.GraphKeys.  The collections to add the
        summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
    # tensor = tf.convert_to_tensor(tensor)
    if summary_op_util.skip_summary():
        return tf.constant("")
    with summary_op_util.summary_scope(name, family, values=[tensor]) as (tag, scope):
          val = tf.py_func(
              py_gif_summary,
              [tag, tensor, max_outputs, fps],
              tf.string,
              stateful=False,
              name=scope)
          summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
    return val


### Numpy functions ##################################################################################
def save_img(out_path, img):
    img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    cv.imwrite(out_path, img[:,:,::-1])


def printVariable(scope, key=tf.GraphKeys.MODEL_VARIABLES):
    print("Scope %s:" % scope)
    variables_names = [[v.name, v.get_shape().as_list()]
                       for v in tf.get_collection(key, scope=scope)]
    total_sz = 0
    for k in variables_names:
        print("Variable: " + k[0])
        print("Shape: " + str(k[1]))
        total_sz += np.prod(k[1])
    print("total size: %d" % total_sz)


def preexec():
    os.setpgrp()

if FLAGS.checkpoint is None:
    raise ValueError(
        'The checkpoint file is needed to performing the test.')

# Declare the test data reader
inference_data = inference_data_loader(FLAGS)
input_shape = [1, ] + list(inference_data.inputs[0].shape)
output_shape = [1, input_shape[1]*4, input_shape[2]*4, 3]
oh = input_shape[1] - input_shape[1]//8 * 8
ow = input_shape[2] - input_shape[2]//8 * 8
paddings = tf.constant([[0, 0], [0, oh], [0, ow], [0, 0]])
print("input shape:", input_shape)
print("output shape:", output_shape)

# build the graph
inputs_raw = tf.placeholder(
    tf.float32, shape=input_shape, name='inputs_raw')

pre_inputs = tf.Variable(tf.zeros(input_shape),
                            trainable=False, name='pre_inputs')
pre_gen = tf.Variable(tf.zeros(output_shape),
                        trainable=False, name='pre_gen')
pre_warp = tf.Variable(tf.zeros(output_shape),
                        trainable=False, name='pre_warp')

transpose_pre = tf.space_to_depth(pre_warp, 4)
inputs_all = tf.concat((inputs_raw, transpose_pre), axis=-1)
with tf.variable_scope('generator'):
    gen_output = generator_F(inputs_all, 3, reuse=False, FLAGS=FLAGS)
    # Deprocess the images outputed from the model, and assign things for next frame
    with tf.control_dependencies([tf.assign(pre_inputs, inputs_raw)]):
        outputs = tf.assign(pre_gen, deprocess(gen_output))

inputs_frames = tf.concat((pre_inputs, inputs_raw), axis=-1)
with tf.variable_scope('fnet'):
    gen_flow_lr = fnet(inputs_frames, reuse=False)
    gen_flow_lr = tf.pad(gen_flow_lr, paddings, "SYMMETRIC")
    gen_flow = upscale_four(gen_flow_lr*4.0)
    gen_flow.set_shape(output_shape[:-1]+[2])
pre_warp_hi = tf.contrib.image.dense_image_warp(pre_gen, gen_flow)
before_ops = tf.assign(pre_warp, pre_warp_hi)

print('Finish building the network')

# In inference time, we only need to restore the weight of the generator
var_list = tf.get_collection(
    tf.GraphKeys.MODEL_VARIABLES, scope='generator')
var_list = var_list + \
    tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')

weight_initiallizer = tf.train.Saver(var_list)

# Define the initialization operation
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if (FLAGS.output_pre == ""):
    image_dir = FLAGS.output_dir
else:
    image_dir = os.path.join(FLAGS.output_dir, FLAGS.output_pre)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

with tf.Session(config=config) as sess:
    # Load the pretrained model
    sess.run(init_op)
    sess.run(local_init_op)

    print('Loading weights from ckpt model')
    weight_initiallizer.restore(sess, FLAGS.checkpoint)
    max_iter = len(inference_data.inputs)

    srtime = 0
    print('Frame evaluation starts!!')
    for i in range(max_iter):
        input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
        feed_dict = {inputs_raw: input_im}
        t0 = time.time()
        if(i != 0):
            sess.run(before_ops, feed_dict=feed_dict)
        output_frame = sess.run(outputs, feed_dict=feed_dict)
        srtime += time.time()-t0

        if(i >= 5):
            name, _ = os.path.splitext(
                os.path.basename(str(inference_data.paths_LR[i])))
            filename = FLAGS.output_name+'_'+name
            print('saving image %s' % filename)
            out_path = os.path.join(image_dir, "%s.%s" %
                                    (filename, FLAGS.output_ext))
            save_img(out_path, output_frame[0])
        else:  # First 5 is a hard-coded symmetric frame padding, ignored but time added!
            print("Warming up %d" % (5-i))
print("total time " + str(srtime) + ", frame number " + str(max_iter))
