import tensorflow as tf
jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

def get_weight(shape, dtype, name):
  return tf.constant(0.03, shape=shape, dtype=dtype, name=name)

def res_skip(acts, audio, n_channels, i, n_layers, dtype):
  if i < n_layers - 1:
      res_skip_channels = 2*n_channels
  else:
      res_skip_channels = n_channels

  w = get_weight([1, 1, n_channels, res_skip_channels], dtype, name="res_skip_w")
  res_skip = tf.nn.conv2d(acts, filter=w, strides=[1,1,1,1], padding="VALID",
    data_format="NCHW", name="res_skip")

  with jit_scope(): # help very little
    if i < n_layers - 1:
      # slice is faster than split
      audio = tf.add(res_skip[:,:n_channels,:,:], audio, name='res_add_audio')
      skip = res_skip[:,n_channels:,:,:]
    else:  # last layer
      skip = res_skip

  return audio, skip

def add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  in_act = input_a + input_b
  t_act = tf.nn.tanh(in_act[:, :n_channels, :, :])
  s_act = tf.nn.sigmoid(in_act[:, n_channels:, :, :])
  acts = t_act * s_act
  return acts

def conv(audio, dilation, filter, name):
  """
  audio: NCHW
  dilation: int, dilation on W dim only!!

  We manually pad to avoid cudnn bug when dilation is 64 or 128.
  """
  p = tf.constant([ [0,0],  # N
                    [0,0],  # C
                    [0,0],  # H
                    [dilation,dilation], ], name="conv_paddings")  # W
  audio  = tf.pad(audio, p, name="conv_input_padded")
  out = tf.nn.conv2d(audio, filter=filter, strides=[1,1,1,1], dilations=[1, 1, 1, dilation],
    padding="VALID", data_format="NCHW", name=name)
  return out

def wn(audio, spect, in_channels, n_mel_channels, n_layers, n_channels, kernel_size):
  """
  audio: NCHW, C = in_channels
  spect: NCHW, C = n_mel_channels

  An optimization possibility is combine audio and output as one tensor
  to avoid slicing into res and skip

  """
  dtype = audio.dtype

  with tf.variable_scope("start"):
    w = get_weight([1, 1, in_channels, n_channels], dtype=dtype, name='w')
    audio = tf.nn.conv2d(audio, filter=w, strides=[1,1,1,1], padding="VALID", data_format="NCHW", name="audio")

  with tf.variable_scope("dilated_convolution"):
    for i in range(n_layers):
      with tf.variable_scope("layer_" + str(i)):
        dilation = 2 ** i
        w = get_weight([1, kernel_size, n_channels, n_channels*2], dtype=dtype, name='in_w')
        in_layer = conv(audio, dilation, w, "in_layer")

        w = get_weight([1, 1, n_mel_channels, n_channels*2], dtype=dtype, name='cond_w')
        cond_layer = tf.nn.conv2d(spect, filter=w, strides=[1,1,1,1], padding="VALID", data_format="NCHW", name="cond_layer")

        with jit_scope():
          acts = add_tanh_sigmoid_multiply(in_layer, cond_layer, n_channels)

        audio, skip = res_skip(acts, audio, n_channels, i, n_layers, dtype)

        if i == 0:
          output = skip
        else:   # use accumulate is also possible, but not supported by UFF
          output = tf.add(output, skip, name="add_output_skip")

  with tf.variable_scope("end"):
    # Separate log_s and b
    w = get_weight([1, 1, n_channels, in_channels], dtype, name='s_w')
    log_s = tf.nn.conv2d(output, filter=w, strides=[1,1,1,1], padding="VALID", data_format="NCHW", name="dense_s")

    w = get_weight([1, 1, n_channels, in_channels], dtype, name='b_w')
    b = tf.nn.conv2d(output, filter=w, strides=[1,1,1,1], padding="VALID", data_format="NCHW", name="dense_b")

  return log_s, b

def create_wn(dtype, use_placeholder):
  import numpy as np
  in_channels = 2
  n_channels = 512
  n_mel_channels = 80
  n_group = 8
  spect_dim = int(n_mel_channels*n_group)
  n_layers = 8
  kernel_size = 3
  t_dim = int(400 * 256 / n_group)
  tf.reset_default_graph()
  tf.random.set_random_seed(0)
  print("Data type: ", dtype)

  if use_placeholder:
    audio = tf.placeholder(dtype, [1, in_channels, 1, t_dim], name="audio")
    spect = tf.placeholder(dtype, [1, spect_dim,   1, t_dim], name='spect')
    f_dict = { # slows down perf due to mem copy
      audio: np.ones([1, in_channels, 1, t_dim], np.float32),
      spect: np.ones([1, spect_dim, 1, t_dim], np.float32),
    }
  else:
    audio = tf.random.uniform([1, in_channels, 1, t_dim], minval=0, maxval=0.1, dtype=tf.float32, name="audio")
    spect = tf.random.uniform([1, spect_dim,   1, t_dim], minval=0, maxval=0.1, dtype=tf.float32, name='spect')
    audio = tf.cast(audio, dtype)
    spect = tf.cast(spect, dtype)
    f_dict = {}

  log_s, b = wn(audio, spect, in_channels, spect_dim, n_layers, n_channels, kernel_size)

  return [log_s, b], f_dict

def wn_test(dtype=tf.float32):
    import numpy as np
    import time

    out, f_dict = create_wn(dtype, False)

    emp_time = 100000.0
    n_runs = 12
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if False:
      print("Turn on XLA!")
      config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
      for _ in range(n_runs):
        start = time.time()
        s, b = sess.run(out, feed_dict=f_dict)
        this_run = time.time() - start
        emp_time = min(emp_time, this_run)
      print("Min ms: {:.2f}".format(emp_time*1000))
      print("Output: {}".format(s.flatten()[0:128:8]))

      options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      sess.run(out, feed_dict=f_dict, options=options, run_metadata=run_metadata)

    from tensorflow.python.client import timeline
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('wn.tf.fp16.v100.json', 'w') as f:
        f.write(chrome_trace)


if __name__ == "__main__":
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# main(tf.float32)
  wn_test(tf.float16)
