import tensorflow as tf
import numpy as np

def upsample(spect, n_mel_channels=80):
    """
    spect:  [batch, n_mel_channels, 1, frames],  NCHW
    """
    # Padding==Same: H = H1 * stride
    # Padding==Valid: out = (in-1) * stride + k
    k = 1024
    s = 256
    upsample_stride = [1, 1, 1, s]
# batch, _, _, t_dim = tf.shape(spect) # batch = 1 t_dim = 400
    batch = tf.shape(spect)[0] # batch = 1 t_dim = 400
    t_dim = tf.shape(spect)[3] # batch = 1 t_dim = 400
    output_shape = [batch, n_mel_channels, 1, (t_dim-1)*s+k]
    filter = tf.constant(1.0, shape=[1, k, n_mel_channels, n_mel_channels],
      dtype=spect.dtype, name='upsample/filter')
    spect = tf.nn.conv2d_transpose(spect, filter, output_shape=output_shape,
        strides=upsample_stride, padding="VALID", data_format="NCHW", name="upsample")

    # trim conv artifacts. maybe pad spec to kernel multiple
    time_cutoff = k - s
    spect = spect[:, :, 0, :-time_cutoff]

    return spect

def regroup(spect, n_mel_channels, n_group, batch=1):
    """
      input dims:  N C   H W
      output dims: N C*8 H W/8
    """
    n_mel_group = n_mel_channels*n_group
    spect = tf.reshape(spect, [batch, n_mel_channels, -1, n_group])   # 80, t/8, 8
    spect = tf.transpose(spect, [0,2,1,3])
    spect = tf.reshape(spect, [batch, -1, n_mel_group])  # t/8, 80*8
    spect = tf.transpose(spect, [0, 2, 1])   # NCW , [1, 80*8, t/8]
    spect = tf.expand_dims(spect, axis=2, name='spect_4d')
    return spect, n_mel_group

def main():
    input = tf.placeholder(tf.float32, shape=(1, 80, 1, None))  # NCW, W = T
    output = upsample(input, 80)
    output, n_mel_group = regroup(output, 80, 8, 1)

    t_dim = 400
    f_dict = {input: np.ones((1, 80, 1, t_dim))}
    with tf.Session() as sess:
      o = sess.run(output, feed_dict=f_dict)
      print("Output shape: ", o.shape)


if __name__ == "__main__":
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  main()
