import pickle
from copy import deepcopy
import itertools
import datetime
import pybulletgym  # register PyBullet enviroments with open ai gym

import numpy as np
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.optim import Adam

torch.cuda.set_device("cuda:0")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.device('cuda'))
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())

import gym
import time

import utils.core as core
from utils.logx import EpochLogger
import user_config as user_config

def sig(x):
    return 1/(1+np.e^(-x))

def sigmoid(middle, stretch, x):
    return sig((x-middle)/stretch)


def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=500, epochs=300, replay_size=int(1e5), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=0,
        update_after=400, update_every=200, num_test_episodes=10, max_ep_len=500,
        logger_kwargs=dict(), save_freq=1, paths=None):

    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an
           ``act`` method which return a action given a state (Depending on pi module)
            ``pi`` neural net for policy
             ``q1`` neural net which should estimate Q-function
             ``q2``  neural net which should estimate Q-function (see https://spinningup.openai.com/en/latest/algorithms/td3.html chapter clipped double-Q learning )
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    def creat_empty_result_dic(agents_dic, epochs):
        result_dic = {}
        for agents_dic_key in agents_dic.keys():
            result_dic[agents_dic_key] = {}
            for epoch in range(epochs):
                result_dic [agents_dic_key]['epoch_{}'.format(epoch)] = {}

        return result_dic

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data, agent_dic):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = agent_dic['ac'].q1(o, a)
        q2 = agent_dic['ac'].q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = agent_dic['ac'].pi(o2)

            # Target Q-values
            q1_pi_targ = agent_dic['ac_targ'].q1(o2, a2)
            q2_pi_targ = agent_dic['ac_targ'].q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data, agent_dic):
        o = data['obs']
        pi, logp_pi = agent_dic['ac'].pi(o)  # return pi_action, logp_pi
        q1_pi = agent_dic['ac'].q1(o, pi)
        q2_pi = agent_dic['ac'].q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(agent_dic):

        batch = agent_dic['replay_buffer'].sample_batch(batch_size)
        # First run one gradient descent step for Q1 and Q2
        agent_dic['q_optimizer'].zero_grad()
        loss_q, q_info = compute_loss_q(batch, agent_dic)
        loss_q.backward()
        agent_dic['q_optimizer'].step()

        # Record things
        agent_dic['logger'].store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in agent_dic['q_params']:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        agent_dic['pi_optimizer'].zero_grad()
        loss_pi, pi_info = compute_loss_pi(batch, agent_dic)
        loss_pi.backward()
        agent_dic['pi_optimizer'].step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in agent_dic['q_params']:
            p.requires_grad = True

        # Record things
        agent_dic['logger'].store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(agent_dic['ac'].parameters(), agent_dic['ac_targ'].parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, agent_dic, deterministic=False):
        return agent_dic['ac'].act(torch.as_tensor(o, dtype=torch.float32),
                                   deterministic)

    def test_agent(agent_dic, test_env):
        for j in range(num_test_episodes):
            ep_ret_cum=[]
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, agent_dic, True).cpu())
                r = get_reward(o)
                ep_ret += r
                ep_len += 1
            ep_ret_cum.append(ep_ret)
            agent_dic['logger'].store(TestEpRet=ep_ret, TestEpLen=ep_len)
            return np.mean(ep_ret_cum), np.std(ep_ret_cum)



    def get_reward(state):
        if state[0] >= 300:
            print("Car has reached the goal")
        '''
        if state[0] > -0.4:
            return (1 + state[0]) ** 2
        if state[0] < -0.8:
            return (1 + state[0]) ** 2
        '''
        return state[0]

    # def get_guiding_policy(ac, paths):
    #     #loads various libraries to make suggestions which action is useful
    #     ac_10 = torch.load(paths['policies'] / 'actor_critic_model_epoch_10')
    #     ac_1 = torch.load(paths['policies'] / 'actor_critic_model_epoch_1')
    #     ac_no_train_0 = torch.load(paths['policies'] / 'actor_critic_model_no_train_0')
    #     ac_no_train_1 = torch.load(paths['policies'] / 'actor_critic_model_no_train_1')
    #     ac_no_train_2 = torch.load(paths['policies'] / 'actor_critic_model_no_train_2')
    #     policies = {'trainable': ac,
    #                 'ac_10': ac_10,
    #                 'ac_1': ac_1,
    #                 'ac_no_train_0' : ac_no_train_0,
    #                 'ac_no_train_1' : ac_no_train_1,
    #                 'ac_no_train_2' : ac_no_train_2}
    #     return policies

    def get_recommended_action(o, agents_dic, counter_follow_policy, agent_policy_to_follow, deterministic=False):
        """
        get recommended actions from the guiding policies
        :param o:  status of the environment
        :param policies: guiding policies
        :param deterministic:
        :return: actions which are suggested by the guiding policies
        """
        recommended_action = {}
        if counter_follow_policy == 0 or agent_policy_to_follow is None:
            for key, value in agents_dic.items():
                action = value['ac'].act(torch.as_tensor(o, dtype=torch.float32),
                                         deterministic)
                recommended_action[key] = action
        else:
            action = agents_dic[agent_policy_to_follow]['ac'].act(torch.as_tensor(o, dtype=torch.float32),
                                                                  deterministic)
            recommended_action[agent_policy_to_follow] = action

        return recommended_action

    def chose_action(o, recommended_action, agent_dic,
                     count_used_policy, prob_chose_random_policy=0.1):
        """
        chose recommended action which has the highest Q-value or a random action
        :param o: status of the environment
        :param recommended_action: recommended actions from guiding policies
        :param ac: actor critic model which is trained
        :param count_used_policy: dic which count which guiding policies is used how often
        :param prob_chose_random_policy: probability to chose a random guiding policy
        :return: action which is executed
        """
        max_critic_value = -100000000
        chosen_policy = None
        chosen_action = None
        # if np.random.rand() < prob_chose_random_policy:
        #     key_list = []
        #     for key, value in recommended_action.items():
        #         key_list.append(key)
        #     temp = np.random.choice(key_list)
        #     chosen_action = recommended_action[temp]
        #     chosen_policy = 'random'
        # else:
        '''
        critic = []
        policies = []
        actions = []
        for key, value in recommended_action.items():
            critic.append(agent_dic['ac'].q1(torch.as_tensor(o, dtype=torch.float32),torch.as_tensor(value, dtype=torch.float32)))
            policies.append(key)
            actions.append(value)
        '''
        for key, value in recommended_action.items():

            critic =    agent_dic['ac'].q1(torch.as_tensor(o, dtype=torch.float32),
                                        torch.as_tensor(value, dtype=torch.float32))
        
            if max_critic_value < critic:
                max_critic_value = critic
                chosen_policy = key
                chosen_action = value
        count_used_policy[chosen_policy] += 1
        out = chosen_action
        i = 4
        for key, value in recommended_action.items():
            if key == chosen_policy:
              out = torch.cat([out,value],dim=0)
              out = torch.cat([out,value],dim=0)
              out = torch.cat([out,value],dim=0)

            else:
              out = torch.cat([out,value],dim=0)
              i = 7
        out = out.resize(i,4).sum(dim=0).div(i)
        return out, chosen_policy

    def get_count_used_policy(agents_dic):
        """
        built dic which count which guiding policies is used how often
        :param policies:  guiding policies
        :return: dic which count which guiding policies is used how often
        """
        count_used_policy = {}#{'random': 0}
        for key, value in agents_dic.items():
            count_used_policy[key] = 0
        return count_used_policy

    def get_agents(number_agents, observation_space, action_space, **ac_kwargs):
        agents_dic = {}
        for i in range(number_agents):
            logger = EpochLogger(**logger_kwargs)
            # logger.save_config(locals())
            agent_dic = {'ac': actor_critic(observation_space, action_space, **ac_kwargs)}
            agent_dic['q_params'] = itertools.chain(agent_dic['ac'].q1.parameters(), agent_dic['ac'].q2.parameters())
            agent_dic['logger'] = logger
            var_counts = tuple(
                core.count_vars(module) for module in [agent_dic['ac'].pi, agent_dic['ac'].q1, agent_dic['ac'].q2])
            agent_dic['logger'].log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
            # Set up optimizers for policy and q-function
            agent_dic['pi_optimizer'] = Adam(agent_dic['ac'].pi.parameters(), lr=lr)
            agent_dic['q_optimizer'] = Adam(agent_dic['q_params'], lr=lr)
            agent_dic['logger'].setup_pytorch_saver(agent_dic['ac'])
            agent_dic['ac_targ'] = deepcopy(agent_dic['ac'])
            for p in agent_dic['ac_targ'].parameters():
                p.requires_grad = False
            agent_dic['replay_buffer'] = core.ReplayBuffer(obs_dim=observation_space.shape, act_dim=action_space.shape, size=replay_size)

            agents_dic['agent_{}'.format(i)] = agent_dic
        return agents_dic

    def setup():
        ##################################################################################################################
        # start of the control code
        ##################################################################################################################

        torch.manual_seed(seed)
        np.random.seed(seed)

        env, test_env = env_fn(), env_fn()
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        number_agents = 4
        agents_dic = get_agents(number_agents, env.observation_space, env.action_space, **ac_kwargs)

        return agents_dic

    def train(agents_dic):
        # Prepare for interaction with environment
        result_dic = creat_empty_result_dic(agents_dic, epochs)
        start_time = time.time()
        env, test_env = env_fn(), env_fn()
        o, ep_ret, ep_len = env.reset(), 0, 0


        # Main loop: collect experience in env and update/log each epoch
        counter_follow_policy = 0
        agent_policy_to_follow = None
        for epoch in range(epochs):
            print('epoch: ', epoch)
            for agent_dic_key, agent_dic in agents_dic.items():
                print('agent', agent_dic_key)
                count_used_policy = get_count_used_policy(agents_dic)

                for t in range(steps_per_epoch):
                    if t < start_steps and epoch==0:
                        a = env.action_space.sample()
                    else:
                        recommended_action = get_recommended_action(o, agents_dic, counter_follow_policy,
                                                                    agent_policy_to_follow=agent_policy_to_follow)
                        a, agent_policy_to_follow = chose_action(o=o, recommended_action=recommended_action,
                                                                 agent_dic=agent_dic,
                                                                 count_used_policy=count_used_policy,
                                                                 prob_chose_random_policy=1 / (len(agents_dic) + 1))

                        if counter_follow_policy == 0:
                            counter_follow_policy = 3
                        else:
                            counter_follow_policy -= 1
                        a = a.cpu()

                    # Step the env
                    o2, r, d, _ = env.step(a)


                    #r = get_reward(o2)


                    ep_ret += r
                    ep_len += 1

                    #env.render()
                    #
                    # Ignore the "done" signal if it comes from hitting the time
                    # horizon (that is, when it's an artificial terminal signal
                    # that isn't based on the agent's state)
                    d = False if ep_len == max_ep_len else d

                    # Store experience to replay buffer
                    agent_dic['replay_buffer'].store(o, a, r, o2, d)
                    # Super critical, easy to overlook step: make sure to update
                    # most recent observation!
                    o = o2

                    # End of trajectory handling
                    if d or (ep_len == max_ep_len):
                        agent_dic['logger'].store(EpRet=ep_ret, EpLen=ep_len)
                        o, ep_ret, ep_len = env.reset(), 0, 0

                    # Update handling
                    if t >= update_after and t % update_every == 0:
                        for j in range(update_every):
                            update(agent_dic)
                # update net at the end of a epoch
                for j in range(update_every):
                    update(agent_dic)

                # Test the performance of the deterministic version of the agent.
                ep_test_mean,  ep_test_std = test_agent(agent_dic, test_env)
                result_dic[agent_dic_key]['epoch_{}'.format(epoch)]['ep_test_mean_std'] = [ep_test_mean, ep_test_std]
                result_dic[agent_dic_key]['epoch_{}'.format(epoch)]['count_used_policy'] = count_used_policy

            #for key, agent_dic in agents_dic.items():
                # Log info about epoch
                agent_dic['logger'].log_tabular('Agent', agent_dic_key)
                agent_dic['logger'].log_tabular('Epoch', epoch)
                agent_dic['logger'].log_tabular('EpRet', with_min_and_max=True)
                agent_dic['logger'].log_tabular('TestEpRet', with_min_and_max=True)
                agent_dic['logger'].log_tabular('EpLen', average_only=True)
                agent_dic['logger'].log_tabular('TestEpLen', average_only=True)
                agent_dic['logger'].log_tabular('TotalEnvInteracts', t)
                agent_dic['logger'].log_tabular('Q1Vals', with_min_and_max=True)
                agent_dic['logger'].log_tabular('Q2Vals', with_min_and_max=True)
                agent_dic['logger'].log_tabular('LogPi', with_min_and_max=True)
                agent_dic['logger'].log_tabular('LossPi', average_only=True)
                agent_dic['logger'].log_tabular('LossQ', average_only=True)
                agent_dic['logger'].log_tabular('Time', time.time() - start_time)
                agent_dic['logger'].dump_tabular()

                        # save trained model

                        # torch.save(ac, '/paths['policies] / 'actor_critic_model_epoch_{}'.format(epoch))

        for agent_dic_key, agent_dic in agents_dic.items():
            torch.save(agent_dic['ac'],'weights_{}'.format(agent_dic_key))
        now = datetime.datetime.now()
        with open( 'results.pkl' , 'wb') as f:
            pickle.dump(result_dic, f)

    agents_dic = setup()
    train(agents_dic)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BipedalWalker-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()
    paths = user_config.get_paths()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=paths['logger'])

    print(torch.get_num_threads())

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed,
        logger_kwargs=logger_kwargs, paths=paths)
