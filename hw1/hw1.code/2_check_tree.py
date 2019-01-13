from tree import Tree
from tree import Node

myTree = Tree()
#print(myTree.get_root())

n = Node('taste')
n.add_value('o')

p = Node('var')
n.add_value('a')

q = Node('var')
n.add_value('b')

r = Node('var')
r.add_value('c')

s = Node('name')

myTree.add_node(n,myTree.get_root())
print("Traversing the tree after adding 1 node")
myTree.print_tree(myTree.get_root(),0)

myTree.add_node(p,n)
#myTree.add_node(p,myTree.search_node(myTree.get_root(),n.feature,n.value))
print("Traversing the tree after adding 2 nodes")
myTree.print_tree(myTree.get_root(),0)
myTree.add_node(q,n)
myTree.add_node(r,n)

print("Traversing the tree after adding 4 nodes")
myTree.print_tree(myTree.get_root(),0)
myTree.add_node(s,r)

"""
n.add_child(p)
n.add_child(q)
n.add_child(r)
r.add_child(s)
"""

print("Traversing the tree after adding 5 nodes")
myTree.print_tree(myTree.root,0)
