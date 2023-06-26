import os
import time

import algorithms as alg
from rsg_envs import registry
from rsg_envs.utils import DataStream, sprint, ss
from rsg_envs.utils.velocity_tracking import *

# cli arguments
args = make_play_argparser().parse_args()

# device for policy inference
device = alg.default_device(args.device)
verbose = not args.quiet

# get task specification
task = registry.get(
    'aliengo', 'velocity-tracking'
)

# config read from both current and runtime cfg file
cfg = task.load_cfg()
env_cfg = cfg['environment']

# step / fixed / gamepad / random command
cmder = make_commander(args.cmd, env_cfg, verbose)

# create environment from the configuration file
env = task.make_env(
    env_cfg, not args.headless, verbose, seed=args.seed
)
env.load_scaling(args.run)
ob_dim = env.ob_dim
act_dim = env.action_dim

# load pretrained model
actor = alg.GeneralActor.make(
    cfg['architecture']['actor'], ob_dim, act_dim
).to(device).restore(os.path.join(args.run, 'state_dict.pt'))
actor = actor.inference()

reward_sum = 0.
episode_start = 0
control_dt = env_cfg['control_dt']
num_steps = int(1e10 if args.endless else args.seconds / control_dt)
stream = DataStream('PlotJuggler', args.dump)
stream.set_timestamp_offset(time.time())

obs = env.reset()

# simulation loop
for step in range(num_steps):
    start_time = time.time()
    action = actor(obs)

    for substep_info in env.substeps(action):
        stream.publish(substep_info, env.getSimTime())
    obs, rew, done, info = env.poststep()
    info.update({
        'Action': action,
        'CmdVel': env.getCommand(),
    })
    stream.publish(info, env.getSimTime())

    if cmder is not None:
        env.setCommand(cmder.read_command(), cmder.normalized)
    reward_sum += rew
    if done or step == num_steps - 1 or (args.timeout and info['timeout']):
        print(
            'End' if done else 'Timeout',
            f'at {ss.u_}{(step + 1 - episode_start) * control_dt:.2f}{ss.r_} s, '
            f'average reward {ss.u_}{reward_sum / (step + 1 - episode_start):.6f}{ss.r_}'
        )
        obs = env.reset()
        actor.reset()
        reward_sum = 0.0
        episode_start = step + 1
        stream.set_timestamp_offset(time.time())
    end_time = time.time()
    if end_time - start_time < control_dt / args.speed:
        time.sleep(control_dt / args.speed + start_time - end_time)

sprint.table(env.get_reward_summary(), num_cols=2, header='Reward Details')
