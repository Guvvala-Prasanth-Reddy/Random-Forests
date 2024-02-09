from tree.Tree import Tree

class Branch:
    def __init__(self,feature_value , tree:Tree ) :
        self.feature_value = feature_value
        self.tree = tree