# Simple `MuJoCo` usage

What can we get out of `MuJoco`?

```
env = gym.make('Reacher-v2')
obs,info = env.reset()
for tick in range(1000):
    env.render()
    action = policy(obs)
    env.step(action)
```
For those who have run the following code, you are already running `MuJoCo` under the hood. However, `MuJoCo` is not just some engine that simulates some robots. In this repository, we focus on the core functionalities of `MuJoCo` and how we can leverage such information in robot learning tasks through the lens of a Roboticist. 

Contact: sungjoon-choi@korea.ac.kr 
