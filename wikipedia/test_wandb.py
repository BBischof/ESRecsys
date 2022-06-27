from absl import app
from absl import flags
from ml_collections import config_dict
import wandb


FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'Jane Random', 'Your name.')
flags.DEFINE_float('float_flag', 0.1, 'A floating point number.')

def main(argv):
  wandb.init(project="test-project", entity="building-recsys")
  wandb.config.update({"flags" : FLAGS})

  cfg = config_dict.ConfigDict()
  cfg.learning_rate = 1e-6
  cfg.momentum = 0.99

  train_cfg = config_dict.ConfigDict()
  train_cfg.shuffle_buffer = 256
  cfg.train_cfg = train_cfg
  
  wandb.config.update({"config" : dict(cfg)})


if __name__ == '__main__':
  app.run(main)
