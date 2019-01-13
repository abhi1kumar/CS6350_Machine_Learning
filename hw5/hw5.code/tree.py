import copy
import collections
import numpy as np

################################################################################
# class Node
################################################################################
class Node(object):
    def __init__(self, feature, label_flag = False):
        self.feature = feature
        self.label_flag = label_flag #Label node is a node with labels.
        self.value = []
        self.child = []


    def add_value(self, value):		
        self.value.append(value)


    def add_child(self, obj):
        self.child.append(obj)


    def remove_last_child(self, obj):
        del self.child[-1]


    def show_content(self):
        print(self.feature)
        for i in range(len(self.value)):
            print(self.value[i])


################################################################################
# class Tree
################################################################################
class Tree(object):
	def __init__(self):
		self.root = None


	def get_root(self):
		return self.root


	def add_node(self, node, parent):
		if parent is None:
			self.root = copy.deepcopy(node)
		else:
			parent.add_child(node)


	def del_node(self,node,parent):
		if (self.root == node or parent is None):
			self.root = None
		else:
			parent.remove_last_child(node)			


	def search_node(self, root, feature, value):
		if(root is None):
			return None
		if(root.feature == feature and collections.Counter(root.value) == collections.Counter(value)):
				return root
		for i in range(len(root.child)):
			found = self.search_node(root.child[i], feature, value)
			if (found is not None):
				return found


	# Empty tree is -1, Root is depth 0, Root+label= 0 
	def get_depth(self, root):
		if(root is None):
			return -1

		if(root.label_flag == True):
			return -1
		elif(len(root.child) == 0):
			return 0
		else:			
			depth = np.zeros((len(root.child),1),dtype='int')
			for i in range(len(root.child)):
				depth[i] = 1 + self.get_depth(root.child[i])
			return np.max(depth)

			
	def print_tree(self, root, depth):
		if(root is not None):
			if(len(root.child) == 0):
				print('   '*depth + "[" + str(root.feature) + "]")
			else:
				for i in range(len(root.child)):
					print('---'*depth + root.feature + " (" + str(root.value[i]) + ")")
					self.print_tree(root.child[i],depth+1)		
		else:
			return
