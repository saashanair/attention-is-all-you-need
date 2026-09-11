from .model import TransformerArchConfig
from .train import OptimiserConfig, TrainConfig, train


def main():
    cfg = TrainConfig(
        transformer_arch_cfg=TransformerArchConfig(),  # all fields have defaults now
        optim_cfg=OptimiserConfig(),                   # all defaults
        num_epochs=1,        # keep tiny for a first smoke run
        batch_size=128,
    )
    train(cfg)

if __name__ == "__main__":
    main()