from __future__ import division

import nnlib
import numpy as np
import ris_model_base as base
import tensorflow as tf
import tfplus


class FGInner(tfplus.nn.Model):

    def __init__(self, name='fg_inner', **kwargs):
        """
        Args:
            inp_depth

            use_bn
            weight_decay

            cnn_depth
            cnn_pool
            dcnn_depth
            dcnn_pool
            add_skip_conn
            cnn_skip_mask
            dcnn_skip_mask

            num_semantic_classes

            add_orientation
            num_orientation_classes

            frozen
        """
        super(FGInner, self).__init__(name=name)
        self.set_all_options(kwargs)
        pass

    def init_default_options(self):
        self.set_default_option('num_semantic_classes', 1)

    def init_var(self):
        self.init_default_options()
        # Options
        inp_depth = self.get_option('inp_depth')
        use_bn = self.get_option('use_bn')
        wd = self.get_option('weight_decay')

        cnn_depth = self.get_option('cnn_depth')
        cnn_pool = self.get_option('cnn_pool')
        dcnn_depth = self.get_option('dcnn_depth')
        dcnn_pool = self.get_option('dcnn_pool')

        num_semantic_classes = self.get_option('num_semantic_classes')
        add_orientation = self.get_option('add_orientation')
        num_orientation_classes = self.get_option('num_orientation_classes')
        frozen = self.get_option('frozen')

        cnn_nlayers = len(cnn_depth)
        cnn_filter_size = [3] * cnn_nlayers
        cnn_channels = [inp_depth] + cnn_depth
        cnn_act = [tf.nn.relu] * cnn_nlayers
        cnn_use_bn = [use_bn] * cnn_nlayers
        if frozen:
            cnn_frozen = [True] * cnn_nlayers
        else:
            cnn_frozen = None

        self.cnn = tfplus.nn.CNN(cnn_filter_size, cnn_channels, cnn_pool,
                                 cnn_act, cnn_use_bn, scope='cnn', wd=wd,
                                 frozen=cnn_frozen)

        # De-ConvNet
        dcnn_nlayers = len(dcnn_depth)
        dcnn_filter_size = [3] * dcnn_nlayers
        dcnn_act = [tf.nn.relu] * (dcnn_nlayers - 1) + [None]
        if frozen:
            dcnn_frozen = [True] * dcnn_nlayers
        else:
            dcnn_frozen = None
        if add_orientation:
            if dcnn_depth[-1] != num_semantic_classes + num_orientation_classes:
                self.log.warning(
                    'Changing the last layer channel size to {}'.format(
                        num_semantic_classes + num_orientation_classes))
                dcnn_depth[-1] = num_semantic_classes + num_orientation_classes
                pass
            pass
        else:
            if dcnn_depth[-1] != num_semantic_classes:
                self.log.warning('Changing the last layer channel size to 1')
                dcnn_depth[-1] = num_semantic_classes
                pass
            pass
        dcnn_channels = [cnn_channels[-1]] + dcnn_depth
        dcnn_use_bn = [use_bn] * (dcnn_nlayers - 1) + [False]
        dcnn_skip_ch = self.build_skip_channels(cnn_channels)
        self.dcnn = tfplus.nn.DCNN(dcnn_filter_size, dcnn_channels,
                                   dcnn_pool, dcnn_act, dcnn_use_bn,
                                   scope='dcnn',
                                   skip_ch=dcnn_skip_ch, wd=wd,
                                   frozen=dcnn_frozen)
        return self

    def build_skip_channels(self, cnn_channels_all):
        add_skip_conn = self.get_option('add_skip_conn')
        cnn_skip_mask = self.get_option('cnn_skip_mask')
        dcnn_skip_mask = self.get_option('dcnn_skip_mask')

        if add_skip_conn:
            dcnn_skip_ch = [0]
            dcnn_skip = [None]
            cnn_skip_layers = []
            cnn_skip_ch = []
            for sk, ch in zip(cnn_skip_mask, cnn_channels_all):
                if sk:
                    cnn_skip_ch.append(ch)
                    pass
                pass
            counter = len(cnn_skip_ch) - 1
            for sk in dcnn_skip_mask:
                if sk:
                    dcnn_skip_ch.append(cnn_skip_ch[counter])
                    counter -= 1
                    pass
                else:
                    dcnn_skip_ch.append(0)
                    pass
                pass
        else:
            dcnn_skip_ch = None
            pass
        return dcnn_skip_ch

    def build_skip_layers(self, h_cnn_all):
        add_skip_conn = self.get_option('add_skip_conn')
        cnn_skip_mask = self.get_option('cnn_skip_mask')
        dcnn_skip_mask = self.get_option('dcnn_skip_mask')

        if add_skip_conn:
            dcnn_skip_layers = [None]
            cnn_skip_layers = []
            for sk, h in zip(cnn_skip_mask, h_cnn_all):
                if sk:
                    cnn_skip_layers.append(h)
                    pass
                pass
            counter = len(cnn_skip_layers) - 1
            for sk in dcnn_skip_mask:
                if sk:
                    dcnn_skip_layers.append(cnn_skip_layers[counter])
                    counter -= 1
                    pass
                else:
                    dcnn_skip_layers.append(None)
                    pass
                pass
            pass
        else:
            dcnn_skip_layers = None
            pass
        return dcnn_skip_layers

    def build(self, inp):
        self.lazy_init_var()
        results = {}

        cnn_depth = self.get_option('cnn_depth')
        num_semantic_classes = self.get_option('num_semantic_classes')
        add_orientation = self.get_option('add_orientation')
        num_orientation_classes = self.get_option('num_orientation_classes')

        x = inp['x']
        # x =  tf.Print(x, [0.0, tf.reduce_mean(x)])
        phase_train = inp['phase_train']
        h_cnn_last = self.cnn({'input': x, 'phase_train': phase_train})
        # h_cnn_last =  tf.Print(h_cnn_last, [1.0, tf.reduce_mean(h_cnn_last)])
        h_cnn_all = [x] + [self.cnn.get_layer(ii)
                           for ii in xrange(len(cnn_depth) - 1)]
        dcnn_skip_layers = self.build_skip_layers(h_cnn_all)
        h_dcnn_last = self.dcnn({
            'input': h_cnn_last, 'phase_train': phase_train,
            'skip': dcnn_skip_layers})
        # h_dcnn_last =  tf.Print(h_dcnn_last, [2.0, tf.reduce_mean(h_dcnn_last)])

        x_shape = tf.shape(x)
        inp_height = x_shape[1]
        inp_width = x_shape[2]

        if add_orientation:
            y_out = h_dcnn_last[:, :, :, :num_semantic_classes]
            d_out = h_dcnn_last[:, :, :, num_semantic_classes:]
            d_out = tf.nn.softmax(tf.reshape(
                d_out, [-1, num_orientation_classes]))
            d_out = tf.reshape(
                d_out, tf.pack([-1, inp_height, inp_width,
                                num_orientation_classes]))
            results['d_out'] = d_out
            pass
        else:
            y_out = tf.reshape(
                h_dcnn_last, tf.pack([-1, inp_height, inp_width,
                                      num_semantic_classes]))
            pass
        y_out = tf.sigmoid(y_out)
        results['y_out'] = y_out
        y_out = tf.Print(y_out, [3.0, tf.reduce_mean(y_out)])
        return results

    def get_save_var_dict(self):
        results = {}
        self.add_prefix_to('cnn', self.cnn.get_save_var_dict(), results)
        self.add_prefix_to('dcnn', self.dcnn.get_save_var_dict(), results)
        return results
    pass
