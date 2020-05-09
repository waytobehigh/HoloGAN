import importlib
import os
import sys

import tensorflow as tf

DEPLOY = True
if DEPLOY:
    from runpy import run_path
    from argparse import Namespace
    cfg = Namespace(**run_path(sys.argv[1]))
else:
    from configs import config as cfg

OUTPUT_DIR = cfg.output_dir
LOGDIR = os.path.join(OUTPUT_DIR, "log")

os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg.gpu_id)
from model_HoloGAN import HoloGAN, ADD_EMBEDDING
from tools.utils import show_all_variables, load_pb


def main(_):
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    emb_graph = None
    if ADD_EMBEDDING:
        emb_graph = load_pb(str(cfg.emb_tf_model))

    with tf.Session(config=run_config, graph=emb_graph) as sess:
        model = HoloGAN(
            sess,
            emb_graph,
            input_width=cfg.input_width,
            input_height=cfg.input_height,
            output_width=cfg.output_width,
            output_height=cfg.output_height,
            dataset_name=cfg.dataset,
            input_fname_pattern=cfg.input_fname_pattern,
            crop=cfg.crop)

        model.build(cfg.build_func)

        show_all_variables()

        if cfg.train:
            train_func = eval("model." + cfg.train_func)
            train_func()
        else:
            if not model.load(LOGDIR)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            model.sample_HoloGAN()


if __name__ == '__main__':
    tf.app.run()
