import numpy as np
import matplotlib.pyplot as plt


def plt_win_rate_mean():
    path = []
    alg_num = 2
    episode_rewards = [[] for _ in range(alg_num)]
    game_map = '6h_vs_8z'
    """path.append('../result/vdn/' + game_map)
    path.append('../result/qmix/' + game_map)
    path.append('../result/qtran_base/' + game_map)
    path.append('../result/qtran_alt/' + game_map)
    path.append('../result/coma/' + game_map)
    path.append('../result/central_v+commnet/' + game_map)
    path.append('../result/central_v+g2anet/' + game_map)
    path.append('../result/maven/' + game_map)"""
    path.append('../result/ddn/'+game_map)
    path.append('../result/ddn+om/'+game_map)
    num_run = 8
    for i in range(alg_num):
        for j in range(num_run):
            try:
                r = np.load(path[i] + '/episode_rewards_{}.npy'.format(j))
                episode_rewards[i].append(r[:100])
            except:
                continue
    print(np.array(episode_rewards).shape)
    episode_rewards = np.array(episode_rewards).mean(axis=1)

    plt.figure()
    plt.ylim(10, 16)
    plt.plot(range(len(episode_rewards[0])), episode_rewards[0], c='b', label='ddn')
    plt.plot(range(len(episode_rewards[1])), episode_rewards[1], c='r', label='ddn+om')
    """plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='vdn')
    plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='qmix')
    plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='qtran_base')
    plt.plot(range(len(win_rates[3])), win_rates[3], c='y', label='qtran_alt')
    plt.plot(range(len(win_rates[4])), win_rates[4], c='c', label='coma')
    plt.plot(range(len(win_rates[7])), win_rates[7], c='#000000', label='maven')
    plt.plot(range(len(win_rates[5])), win_rates[5], c='#FFA500', label='central_v+commnet')
    plt.plot(range(len(win_rates[6])), win_rates[6], c='m', label='central_v+g2anet')"""

    plt.legend()
    plt.xlabel('episodes * 100')
    plt.ylabel('Episode_rewards')
    plt.savefig('../result/overview_{}.png'.format(game_map))
    plt.show()


if __name__ == '__main__':
    plt_win_rate_mean()