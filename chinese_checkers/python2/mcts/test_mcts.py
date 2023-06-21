import random as rnd
from wrappers import pywrapper as pw

class Node:
    def __init__(self, from_, to_, num_, value_, p_):
        self.fromCell = from_
        self.toCell = to_
        self.numSamples = num_
        self.value = value_
        self.p = p_

    def __repr__(self):
        return {'Action': str(self.fromCell) + '->' + str(self.toCell),
                'Visit Count': str(self.numSamples)}

    def __str__(self):
        return 'Node(move={}->{}, visits={}, value={:.3f}, prior={:.3f}'.format(self.fromCell, self.toCell,
                                                                                self.numSamples, self.value, self.p)


class MCTSNode:
    def __init__(self, move, value=0, p=0):
        self.move = move
        self.visited = False
        self.visit_count = 0
        self.value = value
        self.children = []
        self.p = p
        self.C = 0.99

    def update_node(self, value):
        self.value *= self.visit_count
        self.visit_count += 1
        self.value = (self.value + value) / self.visit_count

    def __repr__(self):
        return {'Action': self.move.from_ + '->' + self.move.to_,
                'Visit Count': self.visit_count,
                'Value': self.value,
                'Prior': self.p}

    def __str__(self):
        return 'Node(move=' + str(self.move.from_) + '->' + str(self.move.to_) + \
                ', value=' + str(self.value) + \
                ', visits=' + str(self.visit_count) + \
                ', prior=' + str(self.p) + ')' 


class MCTS:
    def __init__(self, cc, player, states_queue, inference_queue, process_idx):
        self.cc = cc
        self.player = player
        self.samples = 0
        self.states_queue = states_queue
        self.inference_queue = inference_queue
        self.process_idx = process_idx
        

    def runMCTS(self, state, depth):
        root = MCTSNode(None)
        self.samples = 0
        while self.samples < 100: # do 100 just to check
            value = self.MCTS(root, state)

        candidates = []
        max_num_moves = 0
        stats = []

        for child in root.children:
            stats.append(Node(child.move.getFrom(), child.move.getTo(), child.visit_count, child.value, child.p))
            if child.visit_count > max_num_moves:
                max_num_moves = child.visit_count
                candidates = [child.move]
            elif child.visit_count == max_num_moves:
                candidates.append(child.move)

        return rnd.choice(candidates), stats

    def MCTS(self, root, state, depth=0):
        
        self.samples += 1
        
        if self.cc.Done(state):
            v_s = -1
            root.update_node(-v_s)
            return -v_s

        if not root.visited:
            root.visited = True

            if state.getToMove():
                state_rep = pw.reverse_state(state.getBoard())
            else:
                state_rep = state.getBoard()

            # add the state to the queue
            self.states_queue.put((self.process_idx, state_rep))
            
            # wait for the inference to finish
            # print('MCTS: waiting for inference for pid: ', os.getpid())
            y, policy = self.inference_queue.get(block=True)

            self.expand_node(root, state, depth, y, policy)
            v_s = self.evaluate_state(state, y, policy)
            root.update_node(-v_s)
            return -v_s

        else:
            child = self.select_action(root, depth)
            self.cc.ApplyMove(state, child.move)
            value = self.MCTS(child, state, depth+1)
            self.cc.UndoMove(state, child.move)
            root.update_node(-value)

        return -value
    

    def select_action(self, root, depth):
        child = self.player.select_action(root, depth)
        return child

    def expand_node(self, root, state, depth, y, policy):
        moves, probs = self.player.expand_actions(self.cc, state, depth, y, policy)

        for ind, move in enumerate(moves):
            child_node = MCTSNode(move, p=probs[ind])
            root.children.append(child_node)
            
    # add the state to the queue
    def evaluate_state(self, state, y, policy):
        return self.player.evaluate_state(state, self.cc, y, policy)

    def traverse_tree(self, root):
        if len(root.children):
            for child in root.children:
                self.traverse_tree(child)
        else:
            self.cc.freeMove(root.move)


# if __name__ == '__main__':
#     cc = cw.CCheckers()
#     state = cw.CCState()
#     cc.Reset(state)
#     player = uct.UCTPlayer()
#     agent = MCTS(cc, player)

#     state, stats = agent.runMCTS(state, 0)
#     print(state.getFrom(), " -> ", state.getTo())

#     for move in stats:
#         print(move)
