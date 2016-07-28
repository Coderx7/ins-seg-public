"""
Author: Mengye Ren (mren@cs.toronto.edu)

Run inference on a foreground segmentation network with a list of images.

Input:
A plain text file with each line a path to the input image.

Output:
Foreground and angle map images. Re-indexed according to the line number.
E.g. 000000_fg.png, 000000_angle.png, 000001_fg.png, etc.

Usage:
python fg_eval_list.py  --list     {LIST_FILENAME}     \
                        --output   {OUTPUT_FOLDER}     \
                        --restore  {MODEL_FOLDER}

Example:
python fg_eval_list.py  --list     "kitti_test.txt"    \
                        --output   "kitti_output"      \
                        --restore  "kitti_model"

Required flags:
    --list              Input file path. Plain text file, line separated.
    --output            Output folder path.
    --restore           Trained model folder path.

Optional flags:
    --prefetch          Pre-fetch data batches on CPU.
    --inp_height        Resize input image height, default 128.
    --inp_width         Resize input image width, default 448.
    --batch_size        Inference batch size, default 4.
"""

import cslab_environ

import cv2
import numpy as np
import os
import tensorflow as tf
import tfplus
import tfplus.data.list_image_data_provider
from tfplus.utils import BatchIterator, ConcurrentBatchIterator

from model import fg

tfplus.init('Run inference on a foreground segmentation network')

# Program options
tfplus.cmd_args.add('list', 'str', None)
tfplus.cmd_args.add('output', 'str', None)

tfplus.cmd_args.add('inp_height', 'int', 128)
tfplus.cmd_args.add('inp_width', 'int', 448)
tfplus.cmd_args.add('batch_size', 'int', 4)
tfplus.cmd_args.add(
    'restore', 'str', '/ais/gobi4/mren/results/img-count/fg_kitti')
tfplus.cmd_args.add('prefetch', 'bool', False)

opt = tfplus.cmd_args.make()

if opt['list'] is None:
    raise Exception('Need to specify input list using flag --list.')

# Initialize output folder.
log = tfplus.utils.logger.get()
if opt['output'] is None:
    opt['output'] = os.path.join(opt['restore'], 'output')
    if not os.path.exists(opt['output']):
        os.makedirs(opt['output'])

# Initialize session.
sess = tf.Session()

# Initialize model (CPU inference only).
model = (
    tfplus.nn.model.create_from_main('fg')
    .set_gpu(-1)
    .restore_options_from(opt['restore'])
    .build_eval()
    .restore_weights_from(sess, opt['restore'])
)

# Initialize data.
data = tfplus.data.create_from_main('list_img', fname=opt['list'],
                                    inp_height=opt['inp_height'],
                                    inp_width=opt['inp_width'])
batch_iter = BatchIterator(num=data.get_size(), progress_bar=True, cycle=False,
                           shuffle=False, batch_size=opt['batch_size'],
                           get_fn=data.get_batch_idx)

# Multithreaded data prefetching.
if opt['prefetch']:
    batch_iter = ConcurrentBatchIterator(
        batch_iter, max_queue_size=10, num_threads=10)


class FGPlotter(tfplus.utils.Listener):
    """Plot foreground segmentation."""

    def __init__(self, folder=''):
        self._folder = folder
        pass

    def listen(self, results):
        img_id = results['id']
        y_out = results['y_out']
        orig_height = results['orig_height']
        orig_width = results['orig_width']
        for ii in xrange(len(img_id)):
            resize = (orig_width[ii], orig_height[ii])
            _y = (cv2.resize(y_out[ii], resize) * 255).astype('uint8')
            path = os.path.join(self._folder, '{}_fg.png'.format(img_id[ii]))
            cv2.imwrite(path, _y)
            log.info('Output written to {}'.format(path))
        pass

    pass


class AngleMapPlotter(tfplus.utils.Listener):
    """Plot angle map."""

    def __init__(self, folder=''):
        self._folder = folder
        # Color map for different angle classes (0-7)
        # You can change it to more informative
        self._cw = np.array(
            [[255, 17, 0], [255, 137, 0],  [230, 255, 0],  [34, 255, 0],
             [0, 255, 213],  [0, 154, 255], [9, 0, 255], [255, 0, 255]],
            dtype='uint8')
        pass

    def listen(self, results):
        img_id = results['id']
        d_out = results['d_out']
        orig_height = results['orig_height']
        orig_width = results['orig_width']

        for ii in xrange(len(img_id)):
            resize = (orig_width[ii], orig_height[ii])
            _d = cv2.resize(d_out[ii], resize, interpolation=cv2.INTER_NEAREST)
            _d = np.argmax(_d, -1)
            _d = self._cw[_d.reshape(
                [-1])].reshape([orig_height[ii], orig_width[ii], 3])
            path = os.path.join(
                self._folder, '{}_angle.png'.format(img_id[ii]))
            cv2.imwrite(path, _d)
            log.info('Output written to {}'.format(path))
        pass


# Run eval experiment.
(
    tfplus.experiment.create_from_main('train')

    .set_session(sess)
    .set_model(model)

    .add_runner(
        tfplus.runner.create_from_main('basic')
        .set_name('runner')
        .add_output('y_out')
        .add_output('d_out')
        .add_listener(FGPlotter(folder=opt['output']))
        .add_listener(AngleMapPlotter(folder=opt['output']))
        .set_iter(batch_iter)
        .set_phase_train(False))
).run()

sess.close()
