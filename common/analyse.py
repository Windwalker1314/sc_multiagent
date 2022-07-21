import numpy as np
import matplotlib.pyplot as plt


def plt_win_rate_mean():
    path = []
    alg_num = 1
    win_rates = [[] for _ in range(alg_num)]
    game_map = '5m_vs_6m'
    #path.append('../result/vdn/' + game_map)
    path.append('../result/qmix/' + game_map)
    #path.append('../result/qtran_base/' + game_map)
    #path.append('../result/qtran_alt/' + game_map)
    #path.append('../result/coma/' + game_map)
    #path.append('../result/central_v+commnet/' + game_map)
    #path.append('../result/central_v+g2anet/' + game_map)
    #path.append('../result/ddn/'+game_map)
    #path.append('../result/ddn+om/'+game_map)
    #path.append('../result/dtrans/' + game_map)
    num_run = 8
    for i in range(alg_num):
        for j in range(num_run):
            try:
                r = np.load(path[i] + '/win_rates_{}.npy'.format(j))
                win_rates[i].append(r[:400]) #2*60
            except:
                continue
        win_rates[i] = np.array(win_rates[i]).mean(axis=0)
    win_rates = np.array(win_rates)
    print(win_rates.shape)

    plt.figure()
    plt.ylim(0,1)
    """plt.plot(range(len(episode_rewards[0])), episode_rewards[0], c='b', label='ddn')
    plt.plot(range(len(episode_rewards[1])), episode_rewards[1], c='r', label='ddn+om')
    plt.plot(range(len(episode_rewards[2])), episode_rewards[2], c='g', label='dtrans')"""
    #plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='vdn')
    plt.plot(range(len(win_rates[0])), win_rates[0], c='r', label='qmix')
    """plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='qtran_base')
    plt.plot(range(len(win_rates[3])), win_rates[3], c='y', label='qtran_alt')
    plt.plot(range(len(win_rates[4])), win_rates[4], c='c', label='coma')
    plt.plot(range(len(win_rates[7])), win_rates[7], c='#000000', label='maven')
    plt.plot(range(len(win_rates[5])), win_rates[5], c='#FFA500', label='central_v+commnet')
    plt.plot(range(len(win_rates[6])), win_rates[6], c='m', label='central_v+g2anet')"""

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Episode_rewards')
    plt.savefig('../result/overview_{}.png'.format(game_map))
    plt.show()


if __name__ == '__main__':
    plt_win_rate_mean()