import os
import json
import torch
from grid2op.Runner import Runner

from l2rpn_baselines.Kaist.Kaist import Kaist


def evaluate(env,
             model_name='kaist',
             load_path='data',
             logs_path=None,
             nb_episode=1,
             nb_process=1,
             max_steps=-1,
             verbose=False,
             save_gif=False,
             **kwargs):
    """
    In order to submit a valid basline, it is mandatory to provide a "evaluate" function with the same signature as this one.
    Parameters
    ----------
    env: :class:`grid2op.Environment.Environment`
        The environment on which the baseline will be evaluated.
    load_path: ``str``
        The path where the model is stored. This is used by the agent when calling "agent.load)
    logs_path: ``str``
        The path where the agents results will be stored.
    nb_episode: ``int``
        Number of episodes to run for the assessment of the performance.
        By default it's 1.
    nb_process: ``int``
        Number of process to be used for the assessment of the performance.
        Should be an integer greater than 1. By defaults it's 1.
    max_steps: ``int``
        Maximum number of timestep each episode can last. It should be a positive integer or -1.
        -1 means that the entire episode is run (until the chronics is out of data or until a game over).
        By default it's -1.
    verbose: ``bool``
        verbosity of the output
    save_gif: ``bool``
        Whether or not to save a gif into each episode folder corresponding to the representation of the said episode.
    kwargs:
        Other key words arguments that you are free to use for either building the agent save it etc.
    Returns
    -------
    ``None``
    """

    TEST_DEV_ENVS = ['l2rpn_wcci_2020'] 
    if env.name not in TEST_DEV_ENVS:
        raise RuntimeError("This baseline only works for: {} ".format(TEST_DEV_ENVS)) 

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    with open(os.path.join(load_path, 'param.json'), 'r', encoding='utf-8') as f:
        param = json.load(f)

    # Create the agent
    state_mean = torch.load(os.path.join(load_path, 'mean.pt'), map_location=param['device']).cpu()
    state_std = torch.load(os.path.join(load_path, 'std.pt'), map_location=param['device']).cpu()
    agent = Kaist(env, state_mean, state_std, **param)
    agent.load_model(load_path)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent
                    )

    # you can do stuff with your model here

    # start the runner
    res = runner.run(path_save=None,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=False)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)


if __name__ == "__main__":
    """
    This is a possible implementation of the eval script.
    """
    from grid2op import make

    env = make("l2rpn_wcci_2020")

    evaluate(env,
             model_name="kaist",
             load_path="data",
             logs_path="logs-train",            
             nb_episode=1,
             nb_process=1,
             max_steps=864,
             verbose=False,
             save_gif="basline_result")