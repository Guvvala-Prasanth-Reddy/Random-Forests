from tree.Branch import Branch
from tree.Tree import Tree
class Node(Tree):
    def __init__(self , feature):
        self.feature = feature
        self.branches = []
    def add_branch(self,branch:Branch):
        self.branches.append(branch)
     

