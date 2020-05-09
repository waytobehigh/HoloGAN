from main import *

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height
  if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  emb_graph = load_pb(cfg['emb_tf_model'])
  print("FLAGs " + str(FLAGS.dataset))
  with tf.Session(config=run_config, graph=emb_graph) as sess:
    model = HoloGAN(
        sess,
        emb_graph,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop)

    model.build(cfg['build_func'])

    show_all_variables()
    else:
      if not model.load(LOGDIR)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      model.sample_HoloGAN(FLAGS)


if __name__ == '__main__':
  tf.app.run()
