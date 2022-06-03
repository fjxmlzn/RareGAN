import numpy as np
import functools
import tensorflow as tf
from tensorflow.python.ops import array_ops
if float('.'.join(tf.__version__.split('.')[:2])) < 1.15:
    tfgan = tf.contrib.gan
else:
    import tensorflow_gan as tfgan
if float('.'.join(tf.__version__.split('.')[:2])) < 2:
    run_inception_fn = functools.partial(
        tfgan.eval.run_inception, output_tensor="pool_3:0")
else:
    tf = tf.compat.v1
    def run_inception_fn(tensor): return tfgan.eval.run_inception(tensor)[
        'pool_3']


class Metric(object):
    def __init__(self, sess):
        self._sess = sess
        self._model = None

    def set_model(self, model):
        self._model = model

    def build(self):
        pass

    def load(self):
        pass

    def evaluate(self, step_id, iteration_id):
        raise NotImplementedError


class NearestRealDistance(Metric):
    def __init__(self, real_images, num_gen_images=100,
                 num_real_images=50000,
                 *args, **kwargs):
        super(NearestRealDistance, self).__init__(*args, **kwargs)
        self._real_images = real_images
        self._num_gen_images = num_gen_images
        self._num_real_images = num_real_images

        id_ = np.random.permutation(real_images.shape[0])
        self._real_images = self._real_images[id_[:num_real_images]]

    def evaluate(self, step_id, iteration_id):
        images = self._model.sample_high(self._num_gen_images)
        result = 0.0
        for i in range(self._num_gen_images):
            diff = np.sqrt(np.sum(
                np.square(
                    self._real_images - np.expand_dims(images[i], axis=0)),
                axis=(1, 2, 3)))
            result += np.min(diff)
        return {"nearest_real_distance": result / self._num_gen_images}


class FrechetInceptionDistance(Metric):
    """
    Adapted from:
    https://github.com/tsc2017/Frechet-Inception-Distance/blob/master/fid.py
    """

    def __init__(self, real_images, batch_size=64,
                 num_gen_images=10000, num_real_images=5000,
                 image_min=0, image_max=1,
                 *args, **kwargs):
        super(FrechetInceptionDistance, self).__init__(*args, **kwargs)
        self._batch_size = batch_size
        self._num_gen_images = num_gen_images
        self._num_real_images = num_real_images
        self._image_min = image_min
        self._image_max = image_max

        self._real_images = real_images
        if not (np.min(self._real_images) >= self._image_min and
                np.max(self._real_images) <= self._image_max):
            raise Exception("range of pixels incorrect")

        id_ = np.random.permutation(real_images.shape[0])
        self._real_images = self._real_images[id_[:num_real_images]]
        self._real_images = self._transform_image(self._real_images)

    def build(self):
        self._inception_images_pl = \
            tf.placeholder(
                tf.float32,
                [None, None, None, 3],
                name="fid_images")

        self._activations1_pl = \
            tf.placeholder(
                tf.float32,
                [None, None],
                name="fid_activations1")
        self._activations2_pl = \
            tf.placeholder(
                tf.float32,
                [None, None],
                name="fid_activations2")
        self._fcd = tfgan.eval.frechet_classifier_distance_from_activations(
            self._activations1_pl,
            self._activations2_pl)

    def load(self):
        size = 299
        images = tf.compat.v1.image.resize_bilinear(
            self._inception_images_pl,
            [size, size])
        generated_images_list = array_ops.split(
            images, num_or_size_splits=1)
        activations = tf.map_fn(
            fn=run_inception_fn,
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=8,
            back_prop=False,
            swap_memory=True,
            name="fid_run_classifier")
        self._activations = array_ops.concat(array_ops.unstack(activations), 0)

    def _get_inception_activations(self, inps):
        n_batches = int(np.ceil(float(inps.shape[0]) / self._batch_size))
        act = []
        for i in range(n_batches):
            inp = inps[i * self._batch_size:
                       (i + 1) * self._batch_size] / 255. * 2 - 1
            sub_act = self._sess.run(
                self._activations,
                feed_dict={self._inception_images_pl: inp})
            act.append(sub_act)
        act = np.concatenate(act, axis=0)
        return act

    def _activation2distance(self, act1, act2):
        return self._sess.run(
            self._fcd,
            feed_dict={self._activations1_pl: act1,
                       self._activations2_pl: act2})

    def _transform_image(self, images):
        images = ((images - self._image_min) /
                  (self._image_max - self._image_min))
        images = images * 255.

        if images.shape[3] == 1:
            images = np.concatenate([images] * 3, axis=3)

        return images

    def evaluate(self, step_id, iteration_id):
        images = self._model.sample_high(self._num_gen_images)

        if not (np.min(images) >= self._image_min and
                np.max(images) <= self._image_max):
            raise Exception("range of pixels incorrect")
        images = self._transform_image(images)

        act1 = self._get_inception_activations(self._real_images)
        act2 = self._get_inception_activations(images)
        fid = self._activation2distance(act1, act2)

        return {"fid": fid}
