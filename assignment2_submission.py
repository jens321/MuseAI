import random

items=['pumpkin', 'sugar', 'egg', 'egg', 'red_mushroom', 
        'planks', 'planks']

food_recipes = {'pumpkin_pie': ['pumpkin', 'egg', 'sugar'],
                'pumpkin_seeds': ['pumpkin'],
                'bowl': ['planks', 'planks'],
                'mushroom_stew': ['bowl', 'red_mushroom']}

rewards_map = {'pumpkin': -5, 'egg': -25, 'sugar': -10,
               'pumpkin_pie': 100, 'pumpkin_seeds': -50,
               'red_mushroom': 5, 'planks': -5,
               'bowl': -1, 'mushroom_stew': 100}

def is_solution(reward):
    return reward == 200

def get_curr_state(items):
    return tuple(sorted(((item for item in items))))

def choose_action(curr_state, possible_actions, eps, q_table):
    if(random.random() < eps):
        a = random.randint(0, len(possible_actions) - 1)
        return possible_actions[a]
    else:
        best_actions = sorted(q_table[curr_state].items(), key = (lambda x: -x[1]))
        num_same = 0
        best_score = best_actions[0][1]
        for elem in best_actions:
            if elem[1] == best_score:
                num_same += 1
            else:
                break
        return best_actions[random.randint(0, num_same-1)][0]
