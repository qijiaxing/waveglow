import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from upsample import upsample, regroup
from wn import wn, get_weight

jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

def random_normal(spect, t_dim, n_remaining_channels, sigma):
  if isinstance(t_dim, int):
    # Generate from numpy and save as constant
    z_shape = (1,n_remaining_channels,1,t_dim)
    import numpy as np
    z_np = np.random.normal(0., 1., z_shape).astype(np.float32)
    audio = tf.constant(z_np, dtype, name="z")
  else:
    # Generate at run time
    audio = tf.random_normal(
      [1, n_remaining_channels, 1, t_dim],
      mean=0.0, stddev=1.0, dtype=dtype, name="z")

  if sigma != 1.0:
    audio = audio * sigma
  return audio


def waveglow(spect, params):
    """
    spect:  [batch, n_mel_channels, frames],  NCW
    """
    dtype = spect.dtype
    sigma = params["sigma"]
    n_mel_channels = params["n_mel_channels"]
    wn_channels = params["wn_channels"]
    wn_layers = params["wn_layers"]
    n_flows = params["n_flows"]
    n_early_every = params["n_early_every"]
    n_early_size = params["n_early_size"]
    n_remaining_channels = params["n_group"]

    # Calculate initial audio channels for inference
    for k in range(params["n_flows"]):
      if k % params["n_early_every"] == 0 and k > 0:
        n_remaining_channels = n_remaining_channels - params["n_early_size"]

    batch = tf.shape(spect)[0] # batch = 1
    spect = upsample(spect, n_mel_channels)
    spect, n_mel_group = regroup(spect, n_mel_channels, params["n_group"], batch)
    # spect: NCHW, [1, 640, 1, 12800]

    t_dim = tf.shape(spect)[3]  # 12800
    audio = random_normal(spect, t_dim, n_remaining_channels, sigma)

    for k in reversed(range(n_flows)):
      with tf.variable_scope("flow_"+str(k)):
        n_half = int(n_remaining_channels/2)
        audio_0 = audio[:,:n_half,:,:]
        audio_1 = audio[:,n_half:,:,:]

        log_s, b = wn(audio_0, spect, in_channels=n_half, n_mel_channels=n_mel_group,
          n_layers=wn_layers, n_channels=wn_channels, kernel_size=3)

        with jit_scope():
          audio_1 = (audio_1 - b) / tf.exp(log_s)
          audio = tf.concat(values=[audio_0, audio_1], axis=1)

        # Reverse W after training
        with tf.variable_scope("1x1_invertible_conv"):
          w = get_weight([1, 1, n_remaining_channels, n_remaining_channels], dtype=dtype, name='1x1_inv_conv_w')
          audio = tf.nn.conv2d(audio, filter=w, strides=[1,1,1,1], padding='SAME', data_format='NCHW', name='1x1_inv_conv')

        if k % n_early_every == 0 and k > 0:
          z = random_normal(spect, t_dim, n_early_size, sigma)
          audio = tf.concat(values=(z, audio), axis=1, name="append_z")
          n_remaining_channels = n_remaining_channels + params["n_early_size"]

    audio = tf.squeeze(audio, axis=2)
    audio = tf.transpose(audio, [0,2,1])
    audio = tf.reshape(audio, [batch, -1], name="output_audio")
    return audio

def create_model(fp16, batch=1, out_graph_file):
  import numpy as np
  import config
  params = config.params
  n_mel_channels = params["n_mel_channels"]
  t_dim = 400
  dtype = tf.float16 if fp16 else tf.float32
  tf.reset_default_graph()
  tf.random.set_random_seed(0)

  spect = tf.placeholder(dtype, [None, n_mel_channels, 1, None],  name='spect')  # NCW
  f_dict = {spect: np.ones([batch, n_mel_channels, 1, t_dim])*0.1}
  out = waveglow(spect, params)
  tf.logging.info("Created waveglow model")

  if out_graph_file:
    out_dir = "./"
    tf.logging.info("Save waveglow graph to: ", out_graph_file)
    tf.train.write_graph(
      tf.get_default_graph().as_graph_def(), out_dir, out_graph_file, False)

  return out, f_dict

if __name__ == "__main__":
  fp16 = True
  out_graph_file = None
  out_audio, f_dict = create_waveglow(fp16, 1, out_graph_file)
