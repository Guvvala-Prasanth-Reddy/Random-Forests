from tree.Tree import Tree


class Leaf(Tree) :
    def __init__(self , target):
        self.target = target
        