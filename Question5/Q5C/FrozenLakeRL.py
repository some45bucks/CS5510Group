import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback

NAME = "FrozenLake"

def runFrozenLake(steps):
    seed = np.random.randint(0, 2**31 - 1)
    env = make_vec_env('FrozenLake8x8-v1', n_envs=24, seed=seed)

    seed = np.random.randint(0, 2**31 - 1)
    vec_eval = make_vec_env('FrozenLake8x8-v1', n_envs=4, seed=seed)
    evalCallback = EvalCallback(vec_eval,log_path=f'./Question5/Q5C/data/logs/{NAME}', best_model_save_path=f"./Question5/Q5C/data/best/{NAME}",n_eval_episodes=10, eval_freq=1000, deterministic=True)

    model = A2C("MlpPolicy", env)

    model.learn(total_timesteps=steps, progress_bar=True,callback=evalCallback)

    env_eval = VecVideoRecorder(vec_eval, f'./Question5/Q5C/data/videos', record_video_trigger=lambda x: x == 0, video_length=100, name_prefix=f"{NAME}")

    obs = env_eval.reset()
    model = A2C.load(f"./Question5/Q5C/data/best/{NAME}/best_model.zip")
    for i in range(100):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_eval.step(action)
        env_eval.render("human")

    env_eval.close()