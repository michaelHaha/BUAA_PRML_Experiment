# -*- coding: utf-8 -*-
"""
Created on Sun May 27 22:30:49 2018

@author: michael
"""

# -*- coding:utf-8 -*-
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/release/bin'

import numpy as np
import graphviz as gv


#FILE_PATH
File_Path = r'D:\课程\模式识别与机器学习\experiment3\data\bezdekIris.txt'
#save path
save_path = r'D:\课程\模式识别与机器学习\experiment3\graph'

# Construct global name Converter
def Class_Name_Converter(data):
    if isinstance(data, str):
        return {
                'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2,
            }.get(data,'Error-no-attribute')
    else:
       return {
                0: 'Iris-setosa',
                1: 'Iris-versicolor',
                2: 'Iris-virginica',
            }.get(data,'Error-no-attribute')  
                   
def Attribute_Name_Converter(data):
    return {
            0: 'Sepal Length',
            1: 'Sepal Width',
            2: 'Petal Length',
            3: 'Petal Width',
        }.get(data,'Error-no-attribute')


#load data
def Load_Data(File_path):
    data = []
    
    with open(File_Path, 'r') as f:
        for line in f.readlines():
            line = line.split(sep = ',')
            tmp = [float(i) for i in line[:-1]]
            tmp.append(Class_Name_Converter(line[-1].replace('\n','')))
            data.append(tmp)
    
    return data
    
#Calculate Entropy
def Calc_Entropy(dataset):
    NumInputs = len(dataset)
    Labels = {}
    
    for single_Input in dataset:
        currentLabel = single_Input[-1]
        Labels[currentLabel] = Labels.get(currentLabel,0) +1
    
    Entropy = 0.0
    for key in Labels.keys():
        Probablity = float(Labels[key])/NumInputs
        Entropy -= Probablity * np.log2(Probablity)        
    return Entropy
    
#Calculate Information Gain
def calInformGain(dataset, attribute,  threshold):
    dataset.sort(key = lambda x: x[int(attribute)])
    base_Entropy = Calc_Entropy(dataset)
    # Find the threshold position
    Thres_pos = 0
    while (dataset[Thres_pos][int(attribute)] < threshold):
        Thres_pos += 1
    InformGain = base_Entropy-Calc_Entropy(dataset[:Thres_pos])*Thres_pos/len(dataset)
    InformGain -= Calc_Entropy(dataset[Thres_pos:])*(len(dataset)-Thres_pos)/len(dataset)
    return InformGain

def calculateThreshold(dataset, attribute):
    final_threshold = 0.0
    # Sort data according to attribute
    dataset.sort(key = lambda x : x[attribute])
    threshold_list = [data[attribute] for data in dataset]
    entropy_gain_max = 0
    for i in range(1, len(threshold_list)):
        entropy_gain_current = calInformGain(dataset, attribute, threshold_list[i])
        if entropy_gain_current > entropy_gain_max:
            entropy_gain_max = entropy_gain_current
            final_threshold = np.mean([threshold_list[i], threshold_list[i - 1]])
    
    return final_threshold

# Select attribute with maximum entropy gain and its threshold
def slecAttribute(dataset, attributes):
    attribute_selected = 0
    InformGainMax_attri = 0
    threshold_selected = 0.0
    
    for attribute in attributes: 
        #calculate information gain for this attribute
        threshold = calculateThreshold(dataset, attribute)
        Informgain = calInformGain(dataset, attribute, threshold)
        #select the optimal attributes according to information gain of different attributes then,
        if Informgain > InformGainMax_attri:
            InformGainMax_attri = Informgain
            attribute_selected = attribute
            threshold_selected = threshold
    
    return (attribute_selected, threshold_selected)
    
def majorityClass(classList):
    classCount={}
    max_class = 0
    
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    
    for key in classCount.keys():
        if classCount[max_class] < classCount[key]:
            max_class = key
    return max_class

class DecisionTree():
    # Branch nodes are classes and leaf nodes are integers
    class Node():
        def __init__(self, attribute, threshold, left, right):
            self.attribute = attribute
            self.threshold = threshold
            self.left = left
            self.right = right

    # Construct decision tree with dataset including n-1 attributes and 1 flag
    # Build decision tree recursively
    def __init__(self, dataset):
        attributes = [i for i in range(len(dataset[0]) - 1)]
        self.root = self.growTree(dataset, attributes)
    
    @staticmethod    
    def growTree(dataset, attributes):
        classList = [example[-1] for example in dataset]
        """
          when it is a leaf node:
        """
        #if all classes in classList are the same, return the this class
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        #if there are no attributes to slpit dataset
        if len(attributes) == 0:
            return majorityClass(classList)
        
        """
          when it is a  branch node:
        """
        # Selected attribute with maximum information gain
        (attributeSelected, threshold) = slecAttribute(dataset, attributes)
        # Generate two branches with new dataset and attributes
        dataset_l = [data for data in dataset if data[attributeSelected] < threshold]
        dataset_r = [data for data in dataset if data[attributeSelected] >= threshold]
        # Remove current attribute
#        attributes.remove(attributeSelected)
        if len(dataset_l) > 0:
            # Create sub-tree recursively
            left = DecisionTree.growTree(dataset_l, attributes)
        else:  
            # Assign child node with majority flag of parent node
            left = majorityClass(classList)
        if len(dataset_r) > 0:
            # Create sub-tree recursively
            right = DecisionTree.growTree(dataset_r, attributes)
        else:
            # Assign child node with majority flag of parent node
            right = majorityClass(classList)
        myTree = DecisionTree.Node(attributeSelected, threshold, left, right)
#        attributes.append(attributeSelected)
        return myTree
    
    # Visualize the decision tree
    def visualize(self):
        # Depth first search decision tree, add node/edge to set
        def DFS(root, graph, nodeSet, edgeSetL, edgeSetR):
            assert isinstance(root, DecisionTree.Node)
            title = Attribute_Name_Converter(root.attribute) + '<=' + str(root.threshold) + '?'
            graph.node(title)
            # Plot the left tree
            if isinstance(root.left, DecisionTree.Node):
                DFS(root.left, graph, nodeSet, edgeSetL, edgeSetR)
                title_left = Attribute_Name_Converter(root.left.attribute) + '<=' + str(root.left.threshold) + '?'
                edgeSetL.add((title, title_left))
            else:
                nodeSet.add(Class_Name_Converter(root.left))
                edgeSetL.add((title, Class_Name_Converter(root.left)))
            # Plot the right tree
            if isinstance(root.right, DecisionTree.Node):
                DFS(root.right, graph, nodeSet, edgeSetL, edgeSetR)
                title_right = Attribute_Name_Converter(root.right.attribute) + '<=' + str(root.right.threshold) + '?'
                edgeSetR.add((title, title_right))
            else:
                nodeSet.add(Class_Name_Converter(root.right))
                edgeSetR.add((title, Class_Name_Converter(root.right)))
        # Update styles to graph
        def apply_styles(graph, styles):
            graph.graph_attr.update(('graph' in styles and styles['graph']) or {})
            graph.node_attr.update(('nodes' in styles and styles['nodes']) or {})
            graph.edge_attr.update(('edges' in styles and styles['edges']) or {})
            return graph

        self.graph = gv.Graph(format = 'svg')
        nodeSet = set()
        edgeSetL = set()
        edgeSetR = set()
        DFS(self.root, self.graph, nodeSet, edgeSetL, edgeSetR)
        # Plot all elements in set
        for node in nodeSet:
            self.graph.node(node)
        for node in edgeSetL:
            self.graph.edge(*node, **{'label' : 'Yes'})
        for node in edgeSetR:
            self.graph.edge(*node, **{'label' : 'No'})
        # Choose graph styles
        styles = {
            'graph': {
                'fontname' : 'Helvetica',
                'fontsize': '16',
                'fontcolor': 'black',
                'bgcolor': 'white',
            },
            'nodes': {
                'fontname': 'Lucida Grande',
                'shape': 'egg',
                'fontcolor': 'white',
                'color': 'black',
                'style': 'filled',
                'fillcolor': 'blue',
            },
            'edges': {
                'style': 'dashed',
                'color': 'black',
                'arrowhead': 'normal',
                'arrowsize': '1.0',
                'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'black',
            }
        }
        self.graph = apply_styles(self.graph, styles)
        self.graph.render(save_path)
        
if __name__ == '__main__':
    dataset = Load_Data(File_Path)
    tree = DecisionTree(dataset)
    tree.visualize()
    
