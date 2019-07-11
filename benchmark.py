import tensorflow as tf
import numpy as np
import time
tf.logging.set_verbosity(tf.logging.INFO)

def profile(sess, out, f_dict, json_file):
  from tensorflow.python.client import timeline
  run_metadata = tf.RunMetadata()
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  o = sess.run(out, feed_dict=f_dict, options=options, run_metadata=run_metadata)
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open(json_file, 'w') as f:
    f.write(chrome_trace)

def print_perf(points, emp_time, sample_rate):
  rtf = emp_time / points * sample_rate
  kHz = int(points / emp_time / 1000.0)
  print("Points: ", points)
  print("Min time: {:.3f} s".format(emp_time))
  print("RTF: {:.3f}".format(rtf))
  print("kHz: ", kHz)

def run(fetches, feed_dict, n_runs, timeline_file):
  config = tf.ConfigProto()
  # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    emp_time = 100000.0
    tf.logging.info("To run Waveglow benchmark")
    for _ in range(n_runs):
      start = time.time()
      o = sess.run(fetches, feed_dict=feed_dict)
      this_run = time.time() - start
      emp_time = min(emp_time, this_run)

    if timeline_file:
      profile(sess, fetches, feed_dict, timeline_file)

  return o, emp_time

def benchmark(fp16=False, profile_prefix="", batch=1, n_runs=16):
  sample_rate = 22050.0
  from glow import create_waveglow
  out, f_dict = create_waveglow(fp16, batch, "")

  if profile_prefix == "":
    timeline_file = None
  else:
    if fp16:
      timeline_file = profile_prefix + 'fp16.json'
    else:
      timeline_file = profile_prefix + 'fp32.json'

  o, emp_time = run(out, f_dict, n_runs, timeline_file)

  print("Output: {}".format(o.flatten()[1000:1128:32]))
  print_perf(o.shape[1], emp_time, sample_rate)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--task', default="benchmark", type=str)
  parser.add_argument('--gpu', default=0, type=int)
  parser.add_argument('--profile_prefix', default="", type=str)

  args = parser.parse_args()

  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

  if args.task == 'benchmark':
    # benchmark(fp16=False, profile_prefix=args.profile_prefix, batch=args.batch_size)
    benchmark(fp16=True,  profile_prefix=args.profile_prefix, batch=args.batch_size)
