from tree import Tree
from tree import Node

myTree = Tree()
#print(myTree.get_root())

n = Node('taste',5)
p = Node('var',6)
q = Node('var',7)
r = Node('var',8)
s = Node('name',9)

myTree.add_node(n,myTree.get_root())
print("Traversing the tree after adding 1 node")
myTree.print_tree(myTree.get_root(),0)

myTree.add_node(p,myTree.search_node(myTree.get_root(),n.feature,n.value))
print("Traversing the tree after adding 2 nodes")
myTree.print_tree(myTree.get_root(),0)
myTree.add_node(q,myTree.search_node(myTree.get_root(),n.feature,n.value))
myTree.add_node(r,myTree.search_node(myTree.get_root(),q.feature,q.value))

print("Traversing the tree after adding 4 nodes")
myTree.print_tree(myTree.get_root(),0)
myTree.add_node(s,myTree.search_node(myTree.get_root(),r.feature,r.value))

"""
n.add_child(p)
n.add_child(q)
n.add_child(r)
r.add_child(s)
"""

print("Traversing the tree after adding 5 nodes")
myTree.print_tree(myTree.root,0)
