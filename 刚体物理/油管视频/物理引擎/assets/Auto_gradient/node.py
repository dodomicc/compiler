from typing import Any, Dict, List
from enum import Enum, auto
import numpy as np
OP_ADD = 0
OP_MUL = 0

class TensorNode:
    def __init__(self, content: np.ndarray|List, height:int, width: int):
        self.content = np.array(content,dtype=object)
        self.height = height
        self.width = width
        self.mask = np.arange(width * height)
        
    def get_row(self,i:int|float)->'TensorNode':
        mask = np.isclose(np.floor(self.mask/self.width),float(i))
        content = self.content[mask]
        return TensorNode(content,1,self.width)  
    
    def get_col(self,i:int|float)->'TensorNode':
        mask = np.isclose(self.mask%self.width,float(i))
        content = self.content[mask]
        return TensorNode(content,self.height,1)  
    
    def as_row_vector(self) -> 'TensorNode':
            return TensorNode(self.content,1,self.width * self.height) 
        
    def as_col_vector(self) -> 'TensorNode':
            return TensorNode(self.content,self.width * self.height,1)   
        
    def take_value_as_row_vector(self,start:int,end:int) -> 'TensorNode':
        end  = min(end,self.width*self.height)
        mask = (self.mask>=start) & (self.mask<end)
        content = self.content[mask]
        return TensorNode(content,1,len(content))
    
    def take_value_as_col_vector(self,start:int,end:int) -> 'TensorNode':
        end  = min(end,self.width*self.height)
        mask = (self.mask>=start) & (self.mask<end)
        content = self.content[mask]
        return TensorNode(content,len(content),1)
    
    def get_scalar_elem(self) -> object:
        return self.content[0]
    
    def transpose(self) -> 'TensorNode':
        mask = np.arange(self.width * self.height).reshape(self.height,self.width).transpose().ravel()
        print(mask)
        content = self.content[mask]
        return TensorNode(content,self.width,self.height)
    
    def get_width(self)->int:
        return self.width
    
    def get_height(self)->int:
        return self.height
    
    def get_content(self)->int:
        return self.content
    
    
    
    

    
class ScalarNode:
    def __init__(self, operation_type:int, compute_elems:List['ScalarNode']):
        self.operation_type = operation_type
        self.compute_elems = compute_elems
        self.children :List[ScalarNode] = []
        
    @staticmethod
    def create_new_scalar_node(operation_type:int, compute_elems:List['ScalarNode'])->'ScalarNode':
        new_node = ScalarNode(operation_type,compute_elems)
        for entry in compute_elems:
            entry.add_child(new_node)
        return new_node
    
    def add_child(self, child:'ScalarNode'):
        self.children.append(child)
        
    @staticmethod
    def add(elem1:'ScalarNode',elem2:'ScalarNode')->'ScalarNode':
        node = ScalarNode.create_new_scalar_node(OP_ADD,[elem1,elem2])
        return node
    
    @staticmethod
    def mul(elem1:'ScalarNode',elem2:'ScalarNode')->'ScalarNode':
        node = ScalarNode.create_new_scalar_node(OP_MUL,[elem1,elem2])
        return node
    
    
                
            
        
        

class TensorNodeComp:
    @staticmethod
    def add(tensornode1: TensorNode, tensornode2:TensorNode)->TensorNode:
        width = tensornode1.get_width()
        height = tensornode1.get_height()
        content1 = tensornode1.get_content()
        content2 = tensornode2.get_content()
        content = []
        for i in range(len(content1)):
            content.append(ScalarNode.add(content1[i],content2[i]))
        return TensorNode(np.array(content,dtype=object),height,width)
    




    
    