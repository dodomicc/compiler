import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from typing import *
from utils import *

def sub_sup_text(self:str,down:str,up:str)->str:
    text = f"{self}^{{{up}}}_{{{down}}}"
    return text

def frac_text(down:str,up:str)->str:
    text = f"\\frac{{{up}}}{{{down}}}"
    return text

def bracket_text(content:str)->str:
    text = f"\\left( {content}  \\right)"
    return text

def strs2Text(*input)->str:
    strs = list(input)
    text = ""
    for entry in strs:
        text += entry
    return text

def strs2tex(*input)->Tex:
    strs = list(input)
    text = ""
    for entry in strs:
        text += entry
    return Tex(rf"{text}")