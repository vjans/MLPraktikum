import matplotlib.pyplot as plt
import user_config as user_config
import pickle
import numpy as np

import pandas as pd

def ema(values):
    df = pd.DataFrame({'d':values})
    return df.ewm(span=40).mean()


def learning_curve():
    #with open('/home/jbrugger/PycharmProjects/lea_rl/results/result_dic/result_dic_1_agents_20201118_22-18-49.pkl', 'rb') as f:
    #with open('/home/jbrugger/PycharmProjects/lea_rl/results/result_dic/result_dic_4_agents_20201118_22-52-37.pkl', 'rb') as f:

    with open('results1.pkl','rb') as f:
        one_agent_dic = pickle.load(f)

    with open('results.pkl', 'rb') as f:
        four_agent_dic = pickle.load(f)
    four_agent_dic.update(one_agent_dic)
    visualize_dic = {}
    '''
    for agent_key, agent_value in one_agent_dic.items():
        visualize_dic['single_agent'] = {}
        visualize_dic['single_agent']['epoch'] = np.arange(len(agent_value))
        visualize_dic['single_agent']['avg_return'] = []
        visualize_dic['single_agent']['std_return'] = []
        visualize_dic['single_agent']['legend'] = 'single agent'
        visualize_dic['single_agent']['line_style'] = '-'

        for epoch_key, epoch_value in agent_value.items():
            visualize_dic['single_agent']['avg_return'].append(epoch_value['ep_test_mean_std'][0])
            visualize_dic['single_agent']['std_return'].append(epoch_value['ep_test_mean_std'][1])
       '''
    for agent_key, agent_value in four_agent_dic.items():
        visualize_dic[agent_key] = {}
        visualize_dic[agent_key]['epoch'] = np.arange(len(agent_value))
        visualize_dic[agent_key]['avg_return'] = []
        visualize_dic[agent_key]['std_return'] = []
        visualize_dic[agent_key]['legend'] = agent_key
        visualize_dic[agent_key]['line_style'] = ':'

        for epoch_key, epoch_value in agent_value.items():
            visualize_dic[agent_key]['avg_return'].append(epoch_value['ep_test_mean_std'][0])
            visualize_dic[agent_key]['std_return'].append(epoch_value['ep_test_mean_std'][1])


    fig = plt.figure(figsize=(15, 7))
    ax = fig.subplots()
    for key, value_dic in visualize_dic.items():
        v = ema(value_dic['avg_return'])
        print(value_dic['avg_return'])
        v = v.values.tolist()
        v = [i[0] for i in v]
        print(v)
        print("")
        ax.errorbar(value_dic['epoch'],v, yerr=value_dic['std_return'], label=value_dic['legend'], linestyle = value_dic['line_style'])
    ax.set_ylabel('avg reward')
    ax.set_xlabel('Epoch')
    ax.set_title('Hopper-v0')
    ax.legend()
    plt.savefig('one_and_four_agents_compare.png', bbox_inches='tight', dpi=300)


def count_used_policy_plot():


    with open('results.pkl', 'rb') as f:
        four_agent_dic = pickle.load(f)
    visualize_dic = {}
    for agent_key, agent_value in four_agent_dic.items():
        x = np.arange(len(agent_value))
        y = dict.fromkeys(agent_value['epoch_0']['count_used_policy'].keys())
        for key in y.keys():
            y[key] = []
        for epoch_key, epoch_value in agent_value.items():
            for key in y.keys():
                y[key].append(epoch_value['count_used_policy'][key])

        fig = plt.figure(figsize=(15, 7))
        ax = fig.subplots()

        print("again")
        for key, value in y.items():

            value = ema(value)
            ax.plot(x, value, '-', label=key )

        ax.set_ylabel('Number of uses in epoch')
        ax.set_xlabel('Epoch')
        ax.set_title('Hopper-v0_{}'.format(agent_key))
        ax.legend()
        plt.savefig('count_used_policy_{}.png'.format(agent_key), bbox_inches='tight', dpi=300)



if __name__ == '__main__':
    learning_curve()
    count_used_policy_plot()
    #paths = user_config.get_paths()
    #count_used_policy_plot(paths, y_label= 'Number of uses in epoch', x_label='Epoch')
