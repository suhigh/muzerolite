
from training import *


def ttt_model(
        config: MuZeroConfig,
        network: Network,
        optimizer: tf.keras.optimizers.Optimizer) -> tf.keras.Model:
    unrolled_model = build_unrolled_model(config, network)
    unrolled_model.compile(
        loss={
            config.network_config.UNROLLED_REWARDS: config.reward_config.loss,
            config.network_config.UNROLLED_VALUES: config.value_config.loss,
            config.network_config.UNROLLED_POLICY_LOGITS: tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            config.network_config.UNROLLED_REWARDS: config.reward_config.loss_decay,
            config.network_config.UNROLLED_VALUES: config.value_config.loss_decay,
            config.network_config.UNROLLED_POLICY_LOGITS: 1.0
        },
        optimizer=optimizer,
        steps_per_execution=config.training_config.steps_per_execution,
        # metrics={config.network_config.UNROLLED_REWARDS: 'mse', config.network_config.UNROLLED_VALUES: 'mse', config.network_config.UNROLLED_POLICY_LOGITS:'categorical_accuracy'}
    )
    return unrolled_model


def ttt_train_network(config: MuZeroConfig,
                      network: Network,
                      replay_buffer: ReplayBuffer,
                      unrolled_model: Model,
                      saved_models_path: str,
                      wait_to_play: bool = True,
                      epoch_cnt: int = 0,
                      writer: Optional[tf.summary.SummaryWriter] = None,
                      checkpoint_manager: Optional[tf.train.CheckpointManager] = None) -> Dict[str, List[float]]:

    replay_buffer_loginterval = config.training_config.replay_buffer_loginterval

    dataset = replay_buffer.as_dataset(batch_size=config.training_config.batch_size)

    muzero_callback = MuZeroCallback(network=network,
                                     saved_models_path=saved_models_path,
                                     checkpoint_manager=checkpoint_manager,
                                     wait_to_play=wait_to_play,
                                     epoch_cnt=epoch_cnt)
    callbacks = [muzero_callback]
    if writer:
        loss_logger = LossLoggerCallback(config=config, network=network, writer=writer)
        callbacks.append(loss_logger)
        if replay_buffer_loginterval is not None:
            replay_buffer_callback = ReplayBufferLoggerCallback(
                network=network,
                replay_buffer=replay_buffer,
                replay_buffer_loginterval=replay_buffer_loginterval,
                writer=writer)
            callbacks.append(replay_buffer_callback)

    num_epochs = config.training_config.training_steps // config.training_config.checkpoint_interval
    history = unrolled_model.fit(dataset,
                                 epochs=num_epochs,
                                 steps_per_epoch=config.training_config.checkpoint_interval,
                                 callbacks=callbacks)
    return history.history

