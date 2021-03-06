from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args,get_dmix_args, get_ddn_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args


if __name__ == '__main__':
    for i in range(4,5):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        if args.alg in ["ddn", "dmix","dplex","dtrans"]:
            args = get_ddn_args(args)
        if args.alg in ["dmix", "dplex","dtrans"]:
            args = get_dmix_args(args)
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        env_info["unit_dim"] = env.get_ally_num_attributes()
        env_info["move_feats_dim"] = env.get_obs_move_feats_size()
        env_info["enemy_feats_dim"] = env.get_obs_enemy_feats_size()
        env_info["ally_feats_dim"] = env.get_obs_ally_feats_size()
        env_info["env_own_feats_dim"] = env.get_obs_own_feats_size()
        args.move_feats_dim = env_info["move_feats_dim"]
        args.enemy_feats_dim = env_info["enemy_feats_dim"]
        args.ally_feats_dim = env_info["ally_feats_dim"]
        args.env_own_feats_dim = env_info["env_own_feats_dim"]
        args.unit_dim = env_info["unit_dim"]
        print(env_info)
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
