import random

items=['pumpkin', 'sugar', 'egg', 'egg', 'red_mushroom', 'planks', 'planks']

food_recipes = {'pumpkin_pie': ['pumpkin', 'egg', 'sugar'],
                'pumpkin_seeds': ['pumpkin'],
                'bowl': ['planks', 'planks'],
                'mushroom_stew': ['bowl', 'red_mushroom']}

rewards_map = {'pumpkin': -5, 'egg': -25, 'sugar': -10,
               'pumpkin_pie': 100, 'pumpkin_seeds': -50,
               'red_mushroom': 5, 'planks': -5, 'bowl': -1,
               'mushroom_stew': 100}

def is_solution(reward):
    return reward == 200

def get_curr_state(items):
  return tuple(item[0] for item in sorted(items) for i in range(item[1]))

def choose_action(curr_state, possible_actions, eps, q_table):
    rnd = random.random()
    if (rnd > eps):
      max_reward = max(q_table[curr_state].items(), key=lambda t: t[1])[1]
      better_actions = [entry[0] for entry in q_table[curr_state].items() if entry[1] == max_reward]
      a = random.randint(0, len(better_actions) - 1)
      return better_actions[a]
    else:
      a = random.randint(0, len(possible_actions) - 1)
      return possible_actions[a]
