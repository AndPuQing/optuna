import optuna
from optuna.samplers import GPSampler
from optuna.samplers import RandomSampler
from optuna import create_study
import torch

dim = 10

optuna.logging.disable_default_handler()
# Add stream handler of stdout to show the messages
# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "BOLLMMasterCodeZeroShotTop1"  # Unique identifier of the study.


benchmarks = {}


def register_objective(name):
    def decorator(func):
        benchmarks[name] = func
        return func

    return decorator


@register_objective("Ackley")
def ackley_objective(trial):
    x = torch.tensor(
        [trial.suggest_float(f"x{i}", -32, 32) for i in range(dim)], dtype=torch.float32
    )
    first_term = -20 * torch.exp(-0.2 * torch.sqrt(torch.sum(x**2) / dim))
    second_term = -torch.exp(torch.sum(torch.cos(2 * torch.pi * x)) / dim)
    y = first_term + second_term + 20 + torch.exp(torch.tensor(1.0))
    return y.item()


# @register_objective("Griewangk")
# def griewangk_objective(trial):
#     x = np.array([trial.suggest_float(f"x{i}", -600, 600) for i in range(dim)])
#     number = np.arange(1, dim + 1)
#     first_term = np.sum((x**2) / 4000)
#     second_term = np.prod(np.cos(x / np.sqrt(number)))
#     y = first_term - second_term + 1
#     return y


# @register_objective("Rastrigrin")
# def rastrigrin_objective(trial):
#     x = np.array([trial.suggest_float(f"x{i}", -5.12, 5.12) for i in range(dim)])
#     y = np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
#     return y


# @register_objective("Rosenbrock")
# def rosenbrock_objective(trial):
#     x = np.array([trial.suggest_float(f"x{i}", -2.048, 2.048) for i in range(dim)])
#     Mat1 = x[:-1]
#     Mat2 = x[1:]
#     y = np.sum(100 * (Mat2 - Mat1**2) ** 2 + (1 - Mat1) ** 2)
#     return y


# @register_objective("Schwefel")
# def schwefel_objective(trial):
#     x = np.array([trial.suggest_float(f"x{i}", -500, 500) for i in range(dim)])
#     y = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
#     return y


for benchmark_name, benchmark in benchmarks.items():
    file_path = (
        f"/home/happy/work/BO/BOLLMMasterCodeZeroShotTop1/optuna_journal_storage-"
        + benchmark_name
        + "-29-2.log"
    )
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(file_path),
    )
    study_name = f"{benchmark_name}-"
    study = create_study(
        direction="minimize",
        storage=storage,
        study_name=study_name + "29-2",
        sampler=GPSampler(n_startup_trials=100, independent_sampler=RandomSampler()),
    )
    study.optimize(benchmark, n_trials=200, n_jobs=1, show_progress_bar=False)
