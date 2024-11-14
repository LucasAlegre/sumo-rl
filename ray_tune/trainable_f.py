from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch


def objective(x, a, b):  # Define an objective function.
    return a * (x ** 0.5) + b


def trainable(config):  # Pass a "config" dictionary into your trainable.

    for x in range(20):  # "Train" for 20 iterations and compute intermediate scores.
        score = objective(x, config["a"], config["b"])

        train.report({"score": score})  # Send the score to Tune.


def tune_trail():
    # Pass in a Trainable class or function, along with a search space "config".
    tuner = tune.Tuner(trainable,
                       param_space={"a": 2, "b": 4})
    tuner.fit()


def num_samples():
    tuner = tune.Tuner(
        trainable, param_space={"a": 2, "b": 4},
        tune_config=tune.TuneConfig(num_samples=10)
    )
    tuner.fit()


def search_space():
    space = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 1)}
    tuner = tune.Tuner(
        trainable,
        param_space=space,
        tune_config=tune.TuneConfig(num_samples=10)
    )
    tuner.fit()


def bayesopt():
    # Define the search space
    search_space = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 20)}

    # algo = BayesOptSearch(random_search_steps=4)

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            # search_alg=algo,
        ),
        run_config=train.RunConfig(stop={"training_iteration": 50}),
        param_space=search_space,
    )
    tuner.fit()


def tune_scheduler():
    from ray.tune.schedulers import HyperBandScheduler

    # Create HyperBand scheduler and minimize the score
    hyperband = HyperBandScheduler(metric="score", mode="max")

    config = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 1)}

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            num_samples=20,
            scheduler=hyperband,
        ),
        param_space=config,
    )
    tuner.fit()


def tune_result():
    config = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 1)}
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            search_alg=BayesOptSearch(random_search_steps=4),
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": 20},
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result()  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.path  # Get best trial's result directory
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    best_metrics = best_result.metrics  # Get best trial's last results
    best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

if __name__ == "__main__":
    # tune_trail()
    num_samples()
    # search_space()
    # tune_scheduler()
    # tune_result()
    # bayesopt()