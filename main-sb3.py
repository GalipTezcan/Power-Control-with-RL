
from env import CustomEnv
from stable_baselines3 import DQN

# Instantiate the env
seed=52
env=CustomEnv(random_state=seed)

model = DQN("MlpPolicy", env,verbose=1,seed=seed,learning_starts=29970,gamma=0.999,exploration_fraction=0.9,exploration_final_eps=0.1)
for i in range(10):
    model.learn(30000)
    model.save("DQN_"+str(i))
records=[]
for i in range(10):
    env=CustomEnv(random_state=seed)
    del model
    model=DQN.load('DQN_'+str(i))
    model.set_env(env)
    max_timesteps_per_episode = 20 # one AMR frame ms.

    for episode_index in range(1000):
        state,_ = env.reset()
        for timestep_index in range(max_timesteps_per_episode):
            # Perform the power control action and observe the new state.
            action = model.predict(state,deterministic=False)
            next_state, reward, terminated, turncated, _ = env.step(action)
            # make next_state the new current state for the next frame.
            state = next_state
            if terminated or turncated:
                break
    records.append(env.retainability_score())                                    
    print(env.retainability_score())

print(records)
#sb3 no test:59, q-table:76, sb3 with test: 
#11.8 200000

#[0.8147405364753071, 0.8147405364753071, 0.8147405364753071, 0.8147405364753071, 0.8147405364753071, 0.8147405364753071, 0.8147405364753071, 0.8147405364753071, 0.8147405364753071, 0.8147405364753071] 30000 10 iter DQN
#[0.41530317613089507, 0.42086752637749125, 0.4163461538461538, 0.41828188561856905, 0.4512591389114541, 0.8346705567712133, 0.8369645042839657, 0.836069590786572, 0.836069590786572, 0.8369645042839657] 30000 10 iter A2C
#[0.628780581776151, 0.7558403969402523, 0.8273838630806846, 0.8362301101591187, 0.8337408312958435, 0.8347488473671438, 0.8341047503045067, 0.8341047503045067, 0.836069590786572, 0.8344743276283618] 30000 10 iter