class State:
    table = 'T'

    def __init__(self, state):
        self.height = 0
        self.width = 0
        self.max_block_width = 1
        self.blocks = []
        self.divider = '   '
        self.parse(state)

    def parse(self, state):
        import re
        state = re.findall(r'\( *\((.*)\) *\)', state)[0]
        blocks = re.split(r'\) *\(', state)
        for block in blocks:
            current_blocks = re.findall(r'on +(\S) +(\S)', block)[0]
            self.put_block_on(current_blocks[0], current_blocks[1])

    def put_block_on(self, block1, block2):
        if len(block1) > self.max_block_width:
            self.max_block_width = len(block1)
        if block2 == self.table:
            self.blocks.append([block1])
            self.width += 1
            if self.height == 0:
                self.height = 1
            return
        for pile in self.blocks:
            if pile[-1] == block2:
                pile.append(block1)
                if len(pile) > self.height:
                    self.height = len(pile)
            if len(block2) > self.max_block_width:
                self.max_block_width = len(block2)
                
    def visualize(self):
        print('\n')
        for h in range(self.height):
            for w in range(self.width):
                try:
                    block = self.blocks[w-1][self.height-1-h]
                except:
                    block = ' '
                print(self._get_block_with_padding(block), end=self.divider)
            print()
        print('-----------------------')

    def _get_block_with_padding(self, block):
        frame = [block[i] if i < len(block) else ' ' for i in range(self.max_block_width)]
        return ''.join(frame)


def visualize_from_input():
    state = input("Enter a state in the form: ((on a T)(on b a)) where T is the table: ")
    State(state).visualize()
    next_ = input("Visualize a new state? (y/n): ")
    if next_ == 'y':
        visualize_from_input()


def visualize_from_file(file_name='states.txt'):
    with open(file_name, 'r') as file:
        line = file.readline()
        while line:
            State(line).visualize()
            line = file.readline()


if __name__ == "__main__":
    print("Sample states:")
    state = '((on a T)(on b a)(on c T)(on d b)(on e c)(on f T))'
    print(f'Input = {state}')
    State(state).visualize()
    print('Sample states from "states.txt":')
    visualize_from_file('states.txt')
    visualize_from_input()

