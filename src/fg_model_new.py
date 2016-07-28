from __future__ import division

import ris_model_base as base
import tensorflow as tf
import tfplus

from fg_inner import FGInner

tfplus.cmd_args.add('fg:inp_depth', 'int', 3)
tfplus.cmd_args.add('fg:padding', 'int', 16)
tfplus.cmd_args.add('fg:cnn_filter_size', 'list<int>',
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
tfplus.cmd_args.add('fg:cnn_depth', 'list<int>',
                    [8, 8, 16, 16, 32, 32, 64, 64, 128, 128])
tfplus.cmd_args.add('fg:cnn_pool', 'list<int>', [1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
tfplus.cmd_args.add('fg:dcnn_filter_size', 'list<int>',
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
tfplus.cmd_args.add('fg:dcnn_depth', 'list<int>',
                    [128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 1])
tfplus.cmd_args.add('fg:dcnn_pool', 'list<int>',
                    [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1])
tfplus.cmd_args.add('fg:add_skip_conn', 'bool', False)
tfplus.cmd_args.add('fg:cnn_skip_mask', 'list<int>',
                    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
tfplus.cmd_args.add('fg:dcnn_skip_mask', 'list<int>',
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
tfplus.cmd_args.add('fg:segm_loss_fn', 'str', 'iou')
tfplus.cmd_args.add('fg:add_orientation', 'bool', False)
tfplus.cmd_args.add('fg:num_orientation_classes', 'int', 8)
tfplus.cmd_args.add('fg:base_learn_rate', 'float', 1e-3)
tfplus.cmd_args.add('fg:learn_rate_decay', 'float', 0.96)
tfplus.cmd_args.add('fg:steps_per_learn_rate_decay', 'int', 5000)
tfplus.cmd_args.add('fg:frozen', 'bool', False)


class FGModel(tfplus.nn.ContainerModel):

    def __init__(self, name='fg_model'):
        super(FGModel, self).__init__(name=name)
        self.register_option('fg:inp_depth')
        self.register_option('fg:padding')
        self.register_option('fg:cnn_filter_size')
        self.register_option('fg:cnn_depth')
        self.register_option('fg:cnn_pool')
        self.register_option('fg:dcnn_filter_size')
        self.register_option('fg:dcnn_depth')
        self.register_option('fg:dcnn_pool')
        self.register_option('fg:add_skip_conn')
        self.register_option('fg:cnn_skip_mask')
        self.register_option('fg:dcnn_skip_mask')
        self.register_option('fg:segm_loss_fn')
        self.register_option('fg:add_orientation')
        self.register_option('fg:num_orientation_classes')
        self.register_option('fg:base_learn_rate')
        self.register_option('fg:learn_rate_decay')
        self.register_option('fg:steps_per_learn_rate_decay')
        self.register_option('fg:frozen')
        self.inner = FGInner()
        self.add_sub_model(self.inner)
        pass

    def init_default_options(self):
        self.set_default_option('fg:use_bn', True)
        self.set_default_option('fg:weight_decay', 5e-5)
        pass

    def build_input(self):
        inp_depth = self.get_option('fg:inp_depth')
        add_orientation = self.get_option('fg:add_orientation')
        num_orientation_classes = self.get_option('fg:num_orientation_classes')
        x = self.add_input_var(
            'x', [None, None, None, inp_depth], 'float')
        y_gt = self.add_input_var(
            'y_gt', [None, None, None, 1], 'float')
        phase_train = self.add_input_var('phase_train', None, 'bool')
        results = {
            'x': x,
            'y_gt': y_gt,
            'phase_train': phase_train
        }
        if add_orientation:
            d_gt = self.add_input_var('d_gt',
                                      [None, None, None,
                                       num_orientation_classes], 'float')
            results['d_gt'] = d_gt
            pass
        return results

    def init_var(self):
        # Options
        inp_depth = self.get_option('fg:inp_depth')
        use_bn = self.get_option('fg:use_bn')
        wd = self.get_option('fg:weight_decay')
        cnn_depth = self.get_option('fg:cnn_depth')
        cnn_pool = self.get_option('fg:cnn_pool')
        dcnn_depth = self.get_option('fg:dcnn_depth')
        dcnn_pool = self.get_option('fg:dcnn_pool')
        add_skip_conn = self.get_option('fg:add_skip_conn')
        cnn_skip_mask = self.get_option('fg:cnn_skip_mask')
        dcnn_skip_mask = self.get_option('fg:dcnn_skip_mask')
        padding = self.get_option('fg:padding')
        add_orientation = self.get_option('fg:add_orientation')
        num_orientation_classes = self.get_option('fg:num_orientation_classes')
        frozen = self.get_option('fg:frozen')

        self.inner.set_option('inp_depth', inp_depth)
        self.inner.set_option('cnn_depth', cnn_depth)
        self.inner.set_option('cnn_pool', cnn_pool)
        self.inner.set_option('dcnn_depth', dcnn_depth)
        self.inner.set_option('dcnn_pool', dcnn_pool)
        self.inner.set_option('add_skip_conn', add_skip_conn)
        self.inner.set_option('cnn_skip_mask', cnn_skip_mask)
        self.inner.set_option('dcnn_skip_mask', dcnn_skip_mask)
        self.inner.set_option('add_orientation', add_orientation)
        self.inner.set_option('num_orientation_classes',
                              num_orientation_classes)
        self.inner.set_option('frozen', frozen)
        self.inner.set_option('weight_decay', wd)
        self.inner.set_option('use_bn', use_bn)
        self.inner.init_var()
        return self

    def build(self, inp):
        """
        Build the model.

        Args:
            inp: x, phase_train
        """
        self.lazy_init_var()
        results = {}

        cnn_depth = self.get_option('fg:cnn_depth')
        add_orientation = self.get_option('fg:add_orientation')
        num_orientation_classes = self.get_option('fg:num_orientation_classes')

        x = inp['x']
        phase_train = inp['phase_train']
        results['x_trans'] = x
        self.register_var('x_trans', x)

        results = self.inner({'x': x, 'phase_train': phase_train})
        results['x_trans'] = x
        self.register_var('y_out', results['y_out'])
        if add_orientation:
            self.register_var('d_out', results['d_out'])
        return results

    def build_loss(self, inp, output):
        """
        Build the model.

        Args:
            inp: x, phase_train, y_gt, d_gt (orientation mode)
            output: x_trans, y_out, d_out (orientation mode)
        """
        segm_loss_fn = self.get_option('fg:segm_loss_fn')
        add_orientation = self.get_option('fg:add_orientation')
        x = output['x_trans']
        x_shape = tf.shape(x)
        inp_height = x_shape[1]
        inp_width = x_shape[2]
        num_ex = tf.shape(x)[0]
        num_ex_f = tf.to_float(num_ex)
        inp_height_f = tf.to_float(inp_height)
        inp_width_f = tf.to_float(inp_width)
        y_gt = inp['y_gt']
        phase_train = inp['phase_train']
        y_out = output['y_out']
        if add_orientation:
            d_gt = inp['d_gt']

        iou_soft = base.f_iou_all(y_out, y_gt)
        self.register_var('iou_soft', iou_soft)
        iou_hard = base.f_iou_all(tf.to_float(y_out > 0.5), y_gt)
        self.register_var('iou_hard', iou_hard)
        bce = tf.reduce_sum(base.f_bce(y_out, y_gt)) / \
            num_ex_f / inp_height_f / inp_width_f
        if segm_loss_fn == 'iou':
            loss = -iou_soft
        elif segm_loss_fn == 'bce':
            loss = bce
        self.register_var('foreground_loss', loss)

        if add_orientation:
            d_out = output['d_out']
            orientation_ce = tf.reduce_sum(base.f_ce(d_out, d_gt) * y_gt) / \
                num_ex_f / inp_height_f / inp_width_f
            loss += orientation_ce
            self.register_var('orientation_ce', orientation_ce)
            correct = tf.equal(tf.argmax(d_out, 3),
                               tf.argmax(d_gt, 3))
            y_gt_mask = tf.squeeze(y_gt)
            orientation_acc = tf.reduce_sum(
                tf.to_float(correct) * y_gt_mask) / tf.reduce_sum(y_gt_mask)
            self.register_var('orientation_acc', orientation_acc)

        self.add_loss(loss)
        total_loss = self.get_loss()
        self.register_var('loss', total_loss)
        return total_loss

    def build_optim(self, loss):
        base_learn_rate = self.get_option('fg:base_learn_rate')
        steps_per_learn_rate_decay = self.get_option(
            'fg:steps_per_learn_rate_decay')
        learn_rate_decay = self.get_option('fg:learn_rate_decay')
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, self.global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        self.register_var('learn_rate', learn_rate)
        eps = 1e-7
        train_step = tf.train.AdamOptimizer(learn_rate, epsilon=eps).minimize(
            loss, global_step=self.global_step)
        # train_step = tf.train.MomentumOptimizer(
        #     learn_rate, momentum=momentum).minimize(
        #     loss, global_step=self.global_step)
        return train_step

    def get_save_var_dict(self):
        results = {}
        if self.has_var('step'):
            results['step'] = self.get_var('step')
        return results

    pass

tfplus.nn.model.register('fg', FGModel)

if __name__ == '__main__':
    tfplus.nn.model.create_from_main('fg').set_name('fg').build_all()
