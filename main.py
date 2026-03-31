import os
import envs
import json
import wandb
import torch
import minari
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from minari import MinariDataset
from vdfine.memory import ReplayBuffer
from envs.utils import collect_data
from vdfine.train import train_backbone
from vdfine.evaluation import evaluate
from vdfine.utils import jsonify


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational DFINE")
    parser.add_argument("--config", type=str, help="path to the config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.run_id = datetime.now().strftime("%Y%m%d_%H%M")

    wandb.init(
        project="Manifolds control",
        name=config.run_name,
        notes=config.notes,
        config=OmegaConf.to_container(config, resolve=True)
    )

    # prepare logging
    save_dir = Path(config.log_dir) / config.run_id
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(config, save_dir / "config.yaml")
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    logger = logging.getLogger(__name__)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # create env and collect data
    env = envs.make(config=config.env)
    logger.info("collecting data ...")
    collect_data(
        env=env,
        data_dir=save_dir / "data",
        num_episodes=config.data.num_episodes,
        action_repeat=config.data.action_repeat,
    )

    # create replay buffers
    dataset = MinariDataset(data=save_dir / "data")
    test_size = int(len(dataset) * config.data.test_ratio)
    train_size = len(dataset) - test_size
    train_data, test_data = minari.split_dataset(dataset=dataset, sizes=[train_size, test_size])
    train_buffer = ReplayBuffer.load_from_minari(dataset=train_data)
    test_buffer = ReplayBuffer.load_from_minari(dataset=test_data)

    # train and save the backbone
    logging.info("training backbone ...")
    encoder, decoder, dynamics_model = train_backbone(
        config=config.train.backbone,
        train_buffer=train_buffer,
        test_buffer=test_buffer,
    )
    torch.save(encoder.state_dict(), save_dir / "encoder.pth")
    torch.save(decoder.state_dict(), save_dir / "decoder.pth")
    torch.save(dynamics_model.state_dict(), save_dir / "dynamics_model.pth")

    # test the model
    eval_results = evaluate(
        eval_config=config.evaluation,
        cost_train_config=config.train.cost,
        env=env,
        dynamics_model=dynamics_model,
        encoder=encoder,
        train_buffer=train_buffer,
        test_buffer=test_buffer
    )

    eval_results = [jsonify(er) for er in eval_results]
    with open(save_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    wandb.save(save_dir / "eval_results.json")

    wandb.finish()
