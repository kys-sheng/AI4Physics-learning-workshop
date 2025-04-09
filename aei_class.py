
from fractions import Fraction
import itertools
import random
import sys
import re 
import numpy as np
import subprocess
import os
import random
import yaml
import regex as re
from IPython.display import Latex
import shutil
from collections import OrderedDict
#from autoeft.io.basis import BasisFile
#import autoeft as ae
from pathlib import Path
import re
import os
import filecmp
import hashlib
import tarfile


# # Global Definition

# - Conventions for tokens and other definitions for sun symmetries
# - Should be a yaml file eventually
# - Note that SU and U should not have "()" brackets for n 


# Set up dictionaries for easy modification in the future.
# Should be inside a YAML file at some point.

# Generic Symbol for Fields
SYMBOL_DICT = { 0             : "S" , 
                1             : "V" , 
                Fraction(1,2) : "F" }
TEXSYMBOL_DICT = { 0          : r"\phi" , 
                1             : "A" , 
                Fraction(1,2) : r"\psi" }

# Allowed N for the group.
ALLOWED_N     = [1, 2, 3]
INDICES_RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

# The dictionary is used to convert the Dynkin labels to the corresponding partition labels.
SU3_DICT =  {   3   : {"DYNKIN": (1,0), "PARTITION":[1]   }     ,   
               -3   : {"DYNKIN": (0,1), "PARTITION":[1,1] }     ,   
                6   : {"DYNKIN": (0,2), "PARTITION":[2,2] }     ,   
               -6   : {"DYNKIN": (2,0), "PARTITION":[2]   }     ,   
                8   : {"DYNKIN": (1,1), "PARTITION":[2,1] }     ,   
                10  : {"DYNKIN": (3,0), "PARTITION":[3]   }     ,   
               -10  : {"DYNKIN": (0,3), "PARTITION":[3,3] }     ,   
                                       }

CONTRACTIONS_TOKEN  = "CONTRACTIONS"
COMMUTATOR_TOKEN    = "COMMUTATOR"
COMMUTATOR_TOKEN_A  = COMMUTATOR_TOKEN+"_A"
COMMUTATOR_TOKEN_B  = COMMUTATOR_TOKEN+"_B"
LORENTZ_TOKEN       = "LORENTZ"

try:
    CONJUGATION_SYMBOL=ae.conjugation_symbol
except:
    CONJUGATION_SYMBOL="+"


LIST_OF_SU3_INVARIAT_3FIELDS = [1, 1, 1], [1, 3, -3], [1, -3, 3], [1, 6, -6], [1, -6, 6], [1, 8, 8], [1, 10, -10], [1, -10, 10], [3, 1, -3], [3, 3, 3], [3, 3, 6], [3, -3, 1], [3, -3, 8], [3, 6, 3], [3, -6, 8], [3, -6, -10], [3, 8, -3], [3, 8, -6], [3, -10, -6], [-3, 1, 3], [-3, 3, 1], [-3, 3, 8], [-3, -3, -3], [-3, -3, -6], [-3, 6, 8], [-3, 6, 10], [-3, -6, -3], [-3, 8, 3], [-3, 8, 6], [-3, 10, 6], [6, 1, -6], [6, 3, 3], [6, -3, 8], [6, -3, 10], [6, 6, 6], [6, -6, 1], [6, -6, 8], [6, 8, -3], [6, 8, -6], [6, 10, -3], [-6, 1, 6], [-6, 3, 8], [-6, 3, -10], [-6, -3, -3], [-6, 6, 1], [-6, 6, 8], [-6, -6, -6], [-6, 8, 3], [-6, 8, 6], [-6, -10, 3], [8, 1, 8], [8, 3, -3], [8, 3, -6], [8, -3, 3], [8, -3, 6], [8, 6, -3], [8, 6, -6], [8, -6, 3], [8, -6, 6], [8, 8, 1], [8, 8, 8], [8, 8, 10], [8, 8, -10], [8, 10, 8], [8, 10, -10], [8, -10, 8], [8, -10, 10], [10, 1, -10], [10, -3, 6], [10, 6, -3], [10, 8, 8], [10, 8, -10], [10, 10, 10], [10, -10, 1], [10, -10, 8], [-10, 1, 10], [-10, 3, -6], [-10, -6, 3], [-10, 8, 8], [-10, 8, 10], [-10, 10, 1], [-10, 10, 8], [-10, -10, -10]
LIST_OF_SU2_INVARIAT_3FIELDS = [1, 1, 1],[1, 2, 2],[1, 3, 3],[1, 4, 4],[1, 5, 5],[2, 1, 2],[2, 2, 1],[2, 2, 3],[2, 3, 2],[2, 3, 4],[2, 4, 3],[2, 4, 5],[2, 5, 4],[3, 1, 3],[3, 2, 2],[3, 2, 4],[3, 3, 1],[3, 3, 3],[3, 3, 5],[3, 4, 2],[3, 4, 4],[3, 5, 3],[3, 5, 5],[4, 1, 4],[4, 2, 3],[4, 2, 5],[4, 3, 2],[4, 3, 4],[4, 4, 1],[4, 4, 3],[4, 4, 5],[4, 5, 2],[4, 5, 4],[5, 1, 5],[5, 2, 4],[5, 3, 3],[5, 3, 5],[5, 4, 2],[5, 4, 4],[5, 5, 1],[5, 5, 3],[5, 5, 5],


BASE_AUTOEFT_OUTPUT_DIR = "/home/ys/YS/uppsala/research/projects/aimb/autoeft-interface/data/"


FIELD_TOKEN         = "FIELD"
DAGGER_TOKEN        = "DAGGER"
DERIVATIVE_TOKEN    = "DERIVATIVE"
GAMMA_TOKEN         = "GAMMA"
SIGMA_TOKEN         = "SIGMA"
SIGMA_BAR_TOKEN     = "SIGMA_BAR"

PARTICLE_ID_TOKEN   = "ID"
DERIVATIVE_ID_TOKEN = "ID"
GAMMA_ID_TOKEN      = "ID"
SIGMA_ID_TOKEN      = "ID"


SPIN_TOKEN          = "SPIN" 
HELICITY_TOKEN      = "HEL"

def particle_id_token(id):
    return [PARTICLE_ID_TOKEN+str(id)]

def derivative_id_token(id):
    return [DERIVATIVE_ID_TOKEN+str(id)]

def gamma_id_token(id):
    return [GAMMA_ID_TOKEN+str(id)]

def sigma_id_token(id):
    return [SIGMA_ID_TOKEN+str(id)]

def spin_token(spin):
    if spin == 0:             return ["SPIN","0"]
    if spin == 1:             return ["SPIN","1"]
    if spin == Fraction(1,2): return ["SPIN","1/2"]
    raise ValueError("Spin not recognized: "+str(spin))

def helicity_token(helicity):
    if helicity == -1/2:        return [HELICITY_TOKEN,"-1/2"]
    if helicity ==  1/2:        return [HELICITY_TOKEN,"1/2"]
    if helicity == -1  :        return [HELICITY_TOKEN,"-1"]
    if helicity ==  1  :        return [HELICITY_TOKEN,"1"]
    raise ValueError("Helicity not recognized: "+str(helicity))



def group_tokens(n,label=None):
    assert isinstance(n,int), "Group must be an integer"
    if n == 1: 
        if label is None: 
            return ["U1"]
        else:
            assert isinstance(label,str), "Label for group must be a string"
            return ["U1",r"_{"+str(label)+r"}"]
    else:
        if label is None: 
            return ["SU"+str(n)]
        else:
            assert isinstance(label,str), "Label for group must be a string"
            return ["SU"+str(n),r"_{"+str(label)+r"}"]

def check_tokens_format():
    print("Checking Tokens Format")
    assert isinstance(particle_id_token(1) ,list),"particle_id_token output needs to be a list "
    print("particle_id_token          : ",particle_id_token(1))
    assert isinstance(derivative_id_token(1) ,list),"derivative_id_token output needs to be a list "
    print("derivative_id_token        : ",derivative_id_token(1))
    assert isinstance(gamma_id_token(1) ,list),"gamma_id_token output needs to be a list "
    print("gamma_id_token             : ",gamma_id_token(1))
    assert isinstance(sigma_id_token(1) ,list),"sigma_id_token output needs to be a list "
    print("sigma_id_token(1)          : ",sigma_id_token(1))
    assert isinstance(spin_token(0) ,list),"spin_token output needs to be a list "
    print("spin_token(0)              : ",spin_token(0))
    assert isinstance(spin_token(1) ,list),"spin_token output needs to be a list "
    print("spin_token(1)              : ",spin_token(1))
    assert isinstance(spin_token(Fraction(1,2)) ,list),"spin_token output needs to be a list "
    print("spin_token(Fraction(1,2))  : ",spin_token(Fraction(1,2)))
    assert isinstance(helicity_token(1/2) ,list),"helicity_token output needs to be a list "
    print("helicity_token(1)          : ",helicity_token(1/2))
    assert isinstance(helicity_token(-1/2) ,list),"helicity_token output needs to be a list "
    print("helicity_token(-1)         : ",helicity_token(-1/2))
    
    assert isinstance(group_tokens(1) ,list),"group_token output needs to be a list "
    print("group_tokens(1)            : ",group_tokens(1))
    assert isinstance(group_tokens(1,"test") ,list),"group_token output needs to be a list "
    print("group_tokens(1,test)            : ",group_tokens(1,"test"))
    assert isinstance(group_tokens(2) ,list),"group_token output needs to be a list "
    print("group_tokens(2)            : ",group_tokens(2))
    assert isinstance(group_tokens(2,"test") ,list),"group_token output needs to be a list "
    print("group_tokens(2,test)            : ",group_tokens(2,"test"))
    assert isinstance(group_tokens(3) ,list),"group_token output needs to be a list "
    print("group_tokens(3)            : ",group_tokens(3))
    assert isinstance(group_tokens(3,"test") ,list),"group_token output needs to be a list "
    print("group_tokens(3,test)            : ",group_tokens(3,"test"))

    print("Tokens Format OK")
    return
    
check_tokens_format()

# # Misc FUnction


def display_latex(instr,withdollar=True):
    if withdollar:
        instr = r"$\LARGE{" + instr + r"}$"
    else:
        instr = r"\LARGE{" + instr + r"}"
    display(Latex(instr))
    return 



# # Symmetry Class



class Symmetry:
    """
    Represents a symmetryof U(1) or SU(N).

    Attributes:
        n (int)                                                 : The dimension of the symmetry group. (1 for U(1), 2 for SU(2), 3 for SU(3), else not implemented yet)
        rep (int or Fraction)                                   : The representation of the symmetry.
        group_label (str, optional)                             : The name of the symmetry group.                                                     Defaults to None.
        partition (list, optional)                              : The partition of the symmetry group.     Only applicable for SU(2) and SU(3) groups. Defaults to None.
        dynkin (int, optional)                                  : The Dynkin number of the symmetry group. Only applicable for SU(2) and SU(3) groups. Defaults to None.
        name (str)                                              : The name of the symmetry.
        tex_name (str)                                          : The LaTeX representation of the symmetry.
        path_name (str)                                         : The path-friendly representation of the symmetry.
        group (str)                                             : The full name of the symmetry group in string
        tokens (list)                                           : The tokens representing the symmetry.
        group_tokens (list)                                     : The tokens representing the symmetry group.

    Methods:
        __init__(self, n, rep, group_label=None, warning=False) : Initializes a Symmetry object.
        __str__(self)                                           : Returns the name of the symmetry.
        __repr__(self)                                          : Returns a string representation of the Symmetry object.
        __eq__(self, other)                                     : Checks if two Symmetry objects are equal.
        get_tokens(self)                                        : Returns the tokens representing the symmetry.
        get_group_tokens(self)                                  : Returns the tokens representing the symmetry group.
        display(self)                                           : Displays the details of the Symmetry object.
        autoeft_input(self, mode="PARTITION")                   : Returns the AutoEFT input for the symmetry.
        get_conjugate(self)                                     : Returns the conjugate of the symmetry.
        show_all_groups()                                       : Static method to show all instances of the Symmetry class.
        sort_groups(group)                                      : Static method to sort the symmetry groups.
        sort_sym(list_of_sym)                                   : Static method to sort a list of Symmetry objects.
    """

    instances       = []
    group_instances = []

    def __init__(self, n, rep, group_label=None,warning=False):
        """
        
        Args:
            n           (int)             : The dimension of the symmetry group. (1 for U(1), 2 for SU(2), 3 for SU(3))
            rep         (int or Fraction) : The representation of the symmetry in conventional naming.
            dynkin_rep  (tuple)           : The representation of the symmetry in dynkin labels.
            group_label  (str, optional)   : The name of the symmetry group. Defaults to None.
            warning (bool, optional)      : Whether to show warnings. Defaults to False.
        """
        assert n in ALLOWED_N   , f"Only {ALLOWED_N} are allowed"
        assert rep is not None  , "Representation cannot be None"

        # Check if the representation is valid for U1
        if n == 1: 
            assert isinstance(rep,(int,Fraction))       , "Invalid representation for U1: Integers or Fractions"
            assert rep != 0                             , "Invalid representation for U1: Non-zero integers or Fractions"
        
        # Check if the representation is valid for SU(N)
        if n >= 2: 
            assert isinstance(rep,int)           , "Invalid representation for SU(N): Integers"
            assert abs(rep) > 1                  , "Invalid representation for SU(N): Cant be 1 or 0"

        # SU(2) specifics
        if n == 2 and rep < 0: 
            if warning: print(f"Note: SU2 representation, {abs(rep)} and its conjugate, {rep} are the same. {abs(rep)} will be used.")
            rep = abs(rep)

        # SU(3) specifics
        if n == 3: 
            try:
                assert rep in SU3_DICT.keys() , f"Invalid representation for SU3: Available representations are {SU3_DICT.keys()}"
            except:
                assert -rep in SU3_DICT.keys() , f"Invalid representation for SU3: Available representations are {SU3_DICT.keys()}"
                if warning: print(f"Note: {rep} not found in {SU3_DICT.keys()} but {abs(rep)} is. {abs(rep)} will be used.")
                rep = abs(rep)
        
        
        # Set the attributes
        self.n           = n
        self.rep         = rep
        self.group_label  = group_label
        self.partition   = None
        self.dynkin      = None
        
        # U1 scenario
        if n == 1:     
            # Set the name
            if group_label == None:
                self.name       = f"U1 : {rep}"
                self.tex_name   = f"U(1):"+r"{\textbf{"+str(rep)+r"}}"
                self.path_name  = f"U1__{str(float(rep))[:7]}"
                self.group_name = "U1"
                self.group_tex  = "U(1)"
            else :
                self.group_label = group_label
                self.name        = f"U1_{{{group_label}}} : {rep}"
                #self.tex_name   = f"U(1)_"+r"{"+str(group_label)+r"}:\textbf{" + str(rep)+r"}"        
                self.tex_name    = r"U(1)_{"+str(group_label)+r"}:\textbf{" + str(rep)+r"}"        
                self.path_name   = f"U1_{group_label}__{str(float(rep))[:7]}"
                self.group_name  = f"{group_label}"
                self.group_tex   = f"U(1)_"+r"{"+group_label+"}"
        
        # SU2 or SU3 scenario
        else: 
            # Get partitions or Dynkin numbers
            if n == 2: 
                self.partition           = [abs(rep)-1]
                self.dynkin              = (abs(rep)-1)                
            elif n == 3:    
                self.partition           = SU3_DICT[rep]["PARTITION"]
                self.dynkin              = SU3_DICT[rep]["DYNKIN"]
            else:
                raise ValueError("Invalid group")
        
                
            # Set the name
            if group_label == None:
                self.name       = f"SU{n} : {rep}"
                self.tex_name   = f"SU({n})"+r":{\textbf{"+str(rep)+r"}}"
                self.path_name  = f"SU{n}__{rep}"
                self.group_name = f"SU{n}"
                self.group_tex  = f"SU({n})"
            else : 
                self.group_label = group_label
                self.name        = f"SU{n}_{{{group_label}}} : {rep}"
                #self.tex_name   = f"SU({n})_"+r"{"+str(group_label)+r"}:\textbf{" + str(rep)+r"}"              
                self.tex_name    = f"SU({n})"+r"_{"+str(group_label)+r"}:\textbf{" + str(rep)+r"}"              
                self.path_name   = f"SU{n}_{group_label}__{rep}"
                self.group_name  = f"{group_label}"
                self.group_tex   = f"SU({n})_"+r"{"+group_label+r"}"
                if not (re.match(r'^[A-Za-z0-9]+(?:\([A-Za-z0-9]+\))?[A-Za-z0-9]*$', self.group_name)): raise ValueError("Invalid group name")
        
        # Add the instance to the list of instances
        Symmetry.instances.append(self)

        # Add the group to the list of group instances
        if self.group_name not in Symmetry.group_instances: 
             Symmetry.group_instances.append(self.group_name)
             Symmetry.group_instances.sort()
        
        assert self.autoeft_input is not None, f"AutoEFT representation is not defined for {self.name}"
        self.get_group_tokens()
        self.get_tokens()
        assert "(" not in self.group_name, f"Group name should not have parenthesis. self.group_name = {self.group_name}"
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Symmetry({self.n}, {self.rep}, {self.group_label})"
    
    def __eq__(self, other):
        if not isinstance(other, Symmetry):
            return False
        return self.n == other.n and self.rep == other.rep and self.group_label == other.group_label
    

    def get_tokens(self):
        tok = self.group_tokens.copy()
        for char in str(self.rep): tok.append(char)
        self.tokens = tok
        return tok
    
    def get_group_tokens(self):
        self.group_tokens = group_tokens(self.n,self.group_label)
        return self.group_tokens

    def display(self):
        (display_latex(r"\mathtt{"+self.tex_name+"}")   )
        (display_latex(r"\mathtt{"+self.group_tex+"}")   )
        
        print("Symmetry       : ",repr(self))
        print("   Group (AutoEFT Syntax) : ",self.group_name )
        print("    Label                 : ",self.group_label )
        print("    Representation        : ",self.rep        )
        
        print("   Name                   : ",self.name       )
        print("     Tex Name             : ",self.tex_name)
        print("     Path Name            : ",self.path_name  )
        
        print("   AutoEFT Input          : ",self.autoeft_input())
        print("     Partition            : ",self.partition  )
        print("     Dynkin               : ",self.dynkin     )
        
        print("   Tokens                 : ",self.tokens     )
        print("     Group Tokens         : ",self.group_tokens)
        
        print("   Conjugate              : ",self.get_conjugate())
        print()

    def autoeft_input(self,mode="PARTITION"):
        if self.n == 1: return str(self.rep)
        else:
            if mode.upper() == "PARTITION": return str(self.partition)
            if mode.upper() == "DYNKIN"   : return str(self.dynkin)
        raise ValueError("Invalid AutoEFT mode")

    def get_conjugate(self):
        if self.n == 1: return Symmetry(self.n,-self.rep,self.group_label)
        if self.n == 2: return Symmetry(self.n,-self.rep,self.group_label)
        if self.n == 3: return Symmetry(self.n,-self.rep,self.group_label)
        raise ValueError("Invalid group")

    @staticmethod
    def show_all_groups():
        """
        Show all instances of the Group class.
        """
        for group in Symmetry.group_instances:
            print(group)
    
    def sort_groups(group):
        su3_group = np.unique([i.group_name for i in group if 3 == i.n]).tolist()
        su2_group = np.unique([i.group_name for i in group if 2 == i.n]).tolist()
        u1_group =  np.unique([i.group_name for i in group if 1 == i.n]).tolist()
        su3_group.sort()
        su2_group.sort()
        u1_group.sort()
        return su3_group + su2_group + u1_group
    
    def sort_sym(list_of_sym):
        out_list_sym = []
        for igroup in Symmetry.sort_groups(list_of_sym):
            out_list_sym += [i for i in list_of_sym if i.group_name == igroup]
        return out_list_sym



 

# # # Field Class


class Field:
    """ 
     Quantum Field.
    
     Attributes:
        spin (float or str)             : The spin of the field.                                                Must be one of [0, 1, 1/2, '1/2', Fraction(1,2)].
        symmetry (list or Symmetry)     : The symmetry properties of the field.                                 Can be a list of Symmetry objects or a single Symmetry object.
        generation                      : The generation of the field.
        helicity                        : The helicity of the field.
        symbol                          : The symbol of the field. Need to follow AutoEFT restrictions.
        name                            : The name of the field. Automatically generate from symbol and symmetries.
        tex_name                        : The LaTeX representation of the field's name.
        tex_hc_name                     : The LaTeX representation of the field's Hermitian conjugate name.
        path_name                       : The path name of the field.
        gen_name                        : The generation name of the field.                             
        conjugate                       : The conjugate of the field.
        anticommute                     : The anticommute property of the field.                        
        anti (bool)                     : Whether the field is an antiparticle or not.
        symbol                          : The symbol of the field.
        qn_tokens                       : The tokens representing the quantum numbers of the field.
        field_tokens                    : The tokens representing the field.
        antiparticle                    : The antiparticle of the field.
 
    """
    def __init__(self, spin        = 0     ,  
                       symmetry    = None  ,
                       generation  = None  ,
                       helicity    = None  ,
                       symbol      = None  ,
                       tex_name    = None  ,
                       tex_hc_name = None  ,
                       path_name   = None  ,
                       gen_name    = None  ,
                       conjugate   = None  ,
                       anticommute = None  ,
                       anti        = False ):
        
        # Check Spin
        assert spin in [0, 1, 1/2, "1/2", Fraction(1,2)], "Invalid spin value. Must be one of [0, 1, 1/2, '1/2', Fraction(1,2)]."
        if (spin == 1/2 or spin == '1/2' or spin == Fraction(1,2)) : spin = Fraction(1,2)
        if spin == 0 : assert helicity is None, "Scalar fields cannot have helicity"
        # Check Symmetry
        if symmetry is not None:
            assert isinstance(symmetry,(list,Symmetry)) , "Symmetry must be a list or a Symmetry object"
            if isinstance(symmetry,Symmetry): symmetry = [symmetry]
            if isinstance(symmetry,list):
                for sym in symmetry: assert isinstance(sym,Symmetry) , "Symmetry must be a list of Symmetry objects"
            
            # A field cant have 2 different representation of a group
            assert len(symmetry) == len(np.unique([i.group_name for i in symmetry])) , "A field cant have 2 different representation of a group"
            self.symmetry = Symmetry.sort_sym(symmetry)
        else:
            self.symmetry = symmetry

        # set attributes
        self.spin        = spin
        self.generation  = generation      # Useless Variable for now       
        self.helicity    = helicity

        self.tex_name    = tex_name
        self.tex_hc_name = tex_hc_name
        self.path_name   = path_name
        self.gen_name    = gen_name        # Useless Variable for now                                 
        
        self.conjugate   = conjugate       # Useless Variable for now (for AutoEFT)                     
        self.anticommute = anticommute     # Useless Variable for now (for AutoEFT)                     
        self.anti        = anti            # Useless Variable for now             
        
        if symbol is None: 
            self.symbol     = self.get_symbol()
            self.tex_symbol = self.get_tex_symbol()
        else             : 
            self.symbol = symbol
            self.tex_symbol = self.get_tex_symbol(symbol)
        
        # Check for AutoEFT compatibility
        assert self.symbol[0].isalpha() , "Symbol must start with an alphabet"
        assert self.symbol[-1].isalnum() or self.symbol[-1] == CONJUGATION_SYMBOL, "Symbol must end with an alphanumeric character"
        assert self.symbol[0] != "L" and self.symbol[0] != "R", "Symbol cannot start with L or R. L and R reserve for helicity"
        
        self.get_qn_tokens()
        self.get_field_tokens()
        self.name_check()
        

    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, value: object) -> bool: #@TODO
        if not isinstance(value, Field): return False
        return self.check_eq_qn(value)
    
    def check_eq_qn(self,field,hel=True,anti=False):
        if self.spin      != field.spin                       : return False
        if self.symmetry  != field.symmetry                   : return False
        if (self.helicity != field.helicity  ) and hel==True  : return False
        if (self.anti     != field.anti      ) and anti==True : return False
        
        return True
        
    def copy(self):
        return Field(spin        = self.spin        ,
                     symmetry    = self.symmetry    ,
                     generation  = self.generation  ,
                     helicity    = self.helicity    ,
                     symbol      = self.symbol      ,
                     tex_name    = self.tex_name    ,
                     tex_hc_name = self.tex_hc_name ,
                     path_name   = self.path_name   ,
                     gen_name    = self.gen_name    ,
                     conjugate   = self.conjugate   ,
                     anticommute = self.anticommute ,
                     anti        = self.anti        )

    def get_symbol(self):
        # Get the symbol
        self.symbol          = SYMBOL_DICT[self.spin]
        
        # Check the helicity 
        if self.spin == Fraction(1,2): 
            if self.helicity == -1/2  : 
                if self.anti == True : self.symbol       += "R"
                else                 : self.symbol       += "L"
            elif self.helicity == 1/2 : 
                if self.anti == True : self.symbol       += "L"
                else                 : self.symbol       += "R"
            else                      : raise ValueError("Helicity must be one of [+1/2,-1/2] for fermions")
        
        if self.anti  == True: self.symbol += "+" 

        return self.symbol
    
    def get_tex_symbol(self,insymbol=None):
        if insymbol is not None: 
            self.tex_symbol = insymbol.replace("p","'")
            self.tex_symbol = insymbol.replace("+","^+")
            self.tex_symbol = r"{"+self.tex_symbol+r"}"
            return self.tex_symbol
        # Get the symbol
        self.tex_symbol      = TEXSYMBOL_DICT[self.spin]
        
        # Check the helicity 
        if self.spin == Fraction(1,2)  : 
            if self.helicity == -1/2 : 
                if self.anti == True :   self.tex_symbol   += "_R"
                else                 :   self.tex_symbol   += "_L"
            elif self.helicity == 1/2  : 
                if self.anti == True :   self.tex_symbol   += "_L"
                else                 :   self.tex_symbol   += "_R"
            else                     :  raise ValueError("Helicity must be one of [+1/2,-1/2] for fermions")
        
        if self.anti  == True: 
            self.tex_symbol += r"^\dagger" 
        self.tex_symbol = r"{"+self.tex_symbol+r"}"

        return self.tex_symbol
        
    def name_check(self,with_groupname=True): # @TODO
        if self.symmetry is not None: self.name = f'{str(self.symbol)}'+"("+", ".join([i.name for i in self.symmetry])+")"
        else                        : self.name = f'{str(self.symbol)}'
        if self.tex_name is None   : 
            #self.tex_name  =  f'{str(self.tex_symbol)}' + r'({' +",".join([i.tex_name for i in self.symmetry]) + r'})'
            if self.symmetry is not None: 
                if with_groupname :   self.tex_name  =  f'{str(self.tex_symbol)}' + r'({' +",".join([i.tex_name for i in self.symmetry]) + r'})'
                else              :   self.tex_name  =  f'{str(self.tex_symbol)}' + r'_{' +",".join([r"\textbf{"+str(i.rep)+r"}" for i in self.symmetry]) + r'}'
            else                  :   self.tex_name  =  f'{str(self.tex_symbol)}'
        return self.name

    
    def get_qn_tokens(self):
        self.qn_tokens = spin_token(self.spin)
        if self.symmetry is not None:
            for i in self.symmetry:
                self.qn_tokens += i.tokens

        if self.helicity                                         : 
            self.qn_tokens += helicity_token(self.helicity)
        return self.qn_tokens
    
    def get_field_tokens(self):
        #if self.anti == True: self.field_tokens = [FIELD_TOKEN, *self.get_qn_tokens(), DAGGER_TOKEN]
        if CONJUGATION_SYMBOL in self.symbol : self.field_tokens = [FIELD_TOKEN, *self.get_qn_tokens(), DAGGER_TOKEN]
        else                                    : self.field_tokens = [FIELD_TOKEN, *self.get_qn_tokens()]

        return self.field_tokens

    def get_antiparticle(self): 
        if self.helicity: antihel = -self.helicity
        else            : antihel = None

        if self.symmetry is not None : anti_symmetry = [i.get_conjugate() for i in self.symmetry]
        else                         : anti_symmetry = None

        if self.symbol[-1] == CONJUGATION_SYMBOL: anti_symbol = self.symbol[:-1]
        else                                       : anti_symbol = self.symbol + CONJUGATION_SYMBOL

        self.antiparticle = Field(spin        = self.spin        ,
                                  symmetry    = anti_symmetry    ,
                                  generation  = self.generation  , # @TODO 
                                  helicity    = antihel          ,
                                  symbol      = anti_symbol      ,
                                  tex_name    = self.tex_hc_name , # @TODO 
                                  tex_hc_name = self.tex_name    , # @TODO 
                                  path_name   = self.path_name   , # @TODO 
                                  gen_name    = self.gen_name    , # @TODO 
                                  conjugate   = self.conjugate   , # @TODO 
                                  anticommute = self.anticommute , # @TODO 
                                  anti        = not self.anti    )
        return self.antiparticle
    
    def get_oppo_hand_partner(self): 
        assert self.spin == Fraction(1,2), "Opposite hand partner is only defined for fermions"
        oppohel             = -self.helicity
        oppo_symbol         = None
        oppo_tex_name       = None
        oppo_tex_hc_name    = None

        if "L" in self.symbol   : oppo_symbol = self.symbol.replace("L","R")
        elif "R" in self.symbol : oppo_symbol = self.symbol.replace("R","L")
        else                    : 
            print("no L or R in symbol:",self.symbol)
            oppo_symbol = self.symbol

        if self.tex_name is not None:
            if "_L" in self.tex_name   : oppo_tex_name = self.tex_name.replace("_L","_R")
            elif r"_{L}" in self.tex_name   : oppo_tex_name = self.tex_name.replace(r"_{L}",r"_{R}")
            elif "_R" in self.tex_name : oppo_tex_name = self.tex_name.replace("_R","_L")
            elif r"_{R}" in self.tex_name : oppo_tex_name = self.tex_name.replace(r"_{R}",r"_{L}")
            else                       : 
                print("no _L or _R in tex_name:",self.tex_name)
                oppo_tex_name = self.tex_name
        if self.tex_hc_name is not None:
            if "_L" in self.tex_hc_name   : oppo_tex_hc_name = self.tex_hc_name.replace("_L","_R")
            elif r"_{L}" in self.tex_hc_name   : oppo_tex_hc_name = self.tex_hc_name.replace(r"_{L}",r"_{R}")
            elif "_R" in self.tex_hc_name : oppo_tex_hc_name = self.tex_hc_name.replace("_R","_L")
            elif r"_{R}" in self.tex_hc_name : oppo_tex_hc_name = self.tex_hc_name.replace(r"_{R}",r"_{L}")
            else                           : 
                print("no _L or _R in tex_hc_name:",self.tex_hc_name)
                oppo_tex_hc_name = self.tex_hc_name
        
        
        self.oppo_hand_partner = Field(spin   = self.spin        ,
                                  symmetry    = self.symmetry    ,
                                  generation  = self.generation  , # @TODO 
                                  helicity    = oppohel          , 
                                  symbol      = oppo_symbol      , 
                                  tex_name    = oppo_tex_name    , 
                                  tex_hc_name = oppo_tex_hc_name ,  
                                  path_name   = self.path_name   , # @TODO 
                                  gen_name    = self.gen_name    , # @TODO 
                                  conjugate   = self.conjugate   , # @TODO 
                                  anticommute = self.anticommute , # @TODO 
                                  anti        = self.anti        ) 
        return self.oppo_hand_partner
    
    def get_kinetic_term(self):
        self.antiparticle   = self.get_antiparticle()
        if self.spin == 0             : return self.get_kinetic_term_scalar()
        if self.spin == 1             : return self.get_kinetic_term_vector()
        if self.spin == Fraction(1,2) : return self.get_kinetic_term_fermion()
        raise ValueError("Invalid spin value. Must be one of [0, 1, 1/2, '1/2', Fraction(1,2)].")
    
    def get_mass_term(self,majorana=None):
        self.antiparticle   = self.get_antiparticle()
        if self.spin == 0             : return self.get_mass_term_scalar()
        if self.spin == 1             : return self.get_mass_term_vector()
        if self.spin == Fraction(1,2) : return self.get_mass_term_fermion(majorana)
        raise ValueError("Invalid spin value. Must be one of [0, 1, 1/2, '1/2', Fraction(1,2)].")
    
    def get_kinetic_term_scalar(self):
        # Get random numbers for the indices
        did1, did2, pid1, pid2 = random.sample(INDICES_RANGE,4)
        kinetic_tokens_list = ["+"]
        
        # D_mu
        kinetic_tokens_list.append(DERIVATIVE_TOKEN)
        if self.symmetry is not None: 
            for sym in self.symmetry: kinetic_tokens_list+= (sym.group_tokens)
        kinetic_tokens_list += derivative_id_token(did1)

        # Phi^dagger
        kinetic_tokens_list += (self.antiparticle.field_tokens)
        kinetic_tokens_list += particle_id_token(pid1)
        
        # D^mu
        kinetic_tokens_list.append(DERIVATIVE_TOKEN)
        if self.symmetry is not None:
            for sym in self.symmetry: kinetic_tokens_list+= (sym.group_tokens)
        kinetic_tokens_list += derivative_id_token(did2)

        # Phi
        kinetic_tokens_list += (self.field_tokens)
        kinetic_tokens_list += particle_id_token(pid2)
        
        # CONTRACTIONS
        kinetic_tokens_list.append(CONTRACTIONS_TOKEN)
        
        # Lorentz
        kinetic_tokens_list.append(LORENTZ_TOKEN)
        kinetic_tokens_list += derivative_id_token(did1)
        kinetic_tokens_list += derivative_id_token(did2)

        # U or SU Groups
        if self.symmetry is not None:
            for sym in self.symmetry: 
                if sym.n == 1: continue
                kinetic_tokens_list += (sym.group_tokens)
                kinetic_tokens_list += (particle_id_token(pid1))
                kinetic_tokens_list += (particle_id_token(pid2))
        
        # the kinetic term should be a list of tokens and the list's elements should be strings
        assert isinstance(kinetic_tokens_list,list)                 , "Kinetic term should be a list of strings"
        assert all([isinstance(i,str) for i in kinetic_tokens_list]), "Kinetic term should be a list of strings"

        # + D_mu Phi^dagger D^mu Phi
        return kinetic_tokens_list
    
    def get_kinetic_term_vector(self):
        #kinetic_tokens_list = ["+"]
        #return kinetic_tokens_list
        raise NotImplementedError("Not implemented yet")
        
    def get_kinetic_term_fermion(self):
        # Get random numbers for the indices
        gid, did, pid1, pid2 = random.sample(INDICES_RANGE,4)

        # + i    
        kinetic_tokens_list = ["+","i"]
    
        # Psi_L^+ / Psi_R
        if self.helicity   ==  Fraction(1,2) : kinetic_tokens_list += (self.field_tokens)
        elif self.helicity == -Fraction(1,2) : kinetic_tokens_list += (self.antiparticle.field_tokens)
        else                                 : raise ValueError("Helicity must be one of [+1/2,-1/2] for fermions")
        kinetic_tokens_list += particle_id_token(pid1)

        # sigma_bar^mu
        kinetic_tokens_list.append(SIGMA_BAR_TOKEN)
        kinetic_tokens_list += sigma_id_token(gid)

        # D_mu
        kinetic_tokens_list.append(DERIVATIVE_TOKEN)
        if self.symmetry is not None:
            for sym in self.symmetry: kinetic_tokens_list+= (sym.group_tokens)
        kinetic_tokens_list += derivative_id_token(did)

        # Psi_L / Psi_R^+
        if self.helicity   ==  Fraction(1,2) : kinetic_tokens_list += (self.antiparticle.field_tokens)
        elif self.helicity == -Fraction(1,2) : kinetic_tokens_list += (self.field_tokens)
        else                                 : raise ValueError("Helicity must be one of [+1/2,-1/2] for fermions")
        kinetic_tokens_list += particle_id_token(pid2)

        # CONTRACTIONS
        kinetic_tokens_list.append(CONTRACTIONS_TOKEN)

        # Lorentz
        kinetic_tokens_list.append(LORENTZ_TOKEN)
        kinetic_tokens_list += sigma_id_token(gid)
        kinetic_tokens_list += derivative_id_token(did)
        
        kinetic_tokens_list.append(LORENTZ_TOKEN)
        kinetic_tokens_list += sigma_id_token(gid)
        kinetic_tokens_list += particle_id_token(pid1)
        kinetic_tokens_list += particle_id_token(pid2)

        # SU Groups
        if self.symmetry is not None:
            for sym in self.symmetry: 
                if sym.n == 1: continue
                kinetic_tokens_list+= (sym.group_tokens)
                kinetic_tokens_list+= particle_id_token(pid1)
                kinetic_tokens_list+= particle_id_token(pid2)

        return kinetic_tokens_list

    def get_mass_term_scalar(self):
        raise KeyError("TODO: need update, bug exist, contractions when there is U1 only. Please fix it!")
        pid1, pid2 = random.sample(INDICES_RANGE,2)

        mass_tokens_list = ["+"]
        
        # Phi^dagger Phi 
        mass_tokens_list += self.antiparticle.field_tokens + particle_id_token(pid1)
        mass_tokens_list += self.field_tokens              + particle_id_token(pid2)

        # U or SU Groups
        if self.symmetry is not None:
            # CONTRACTIONS
            mass_tokens_list.append(CONTRACTIONS_TOKEN)
            for sym in self.symmetry: 
                if sym.n == 1: continue
                mass_tokens_list+= (sym.group_tokens)
                mass_tokens_list += particle_id_token(pid1)
                mass_tokens_list += particle_id_token(pid2)
        
        # + Phi^dagger Phi
        return mass_tokens_list

    def get_mass_term_vector(self):
        raise NotImplementedError("Not implemented yet")

    def get_mass_term_fermion(self,majorana=False,warning=True): #@TODO: NOT DONE
        pid1, pid2 = random.sample(INDICES_RANGE,2)
        pid3, pid4 = random.sample(INDICES_RANGE,2)
        
        
        # First term: (Psi_L Psi_L or Psi_R Psi_R ) or (Psi_L^+ Psi_L^+ or Psi_R^+ Psi_R^+)
        mass_tokens_list = ["-"]
        if majorana:
            # Psi_L Psi_L or Psi_R Psi_R 
            mass_tokens_list += self.field_tokens + particle_id_token(pid1) 
            mass_tokens_list += self.field_tokens + particle_id_token(pid2)
        else:
            self.get_oppo_hand_partner()
            self.oppo_hand_partner.get_antiparticle()

            if self.helicity ==  -Fraction(1,2) : # Psi_L
                # Psi_L Psi_R^+
                mass_tokens_list += self.field_tokens                                + particle_id_token(pid1) # Psi_L
                mass_tokens_list += self.oppo_hand_partner.antiparticle.field_tokens + particle_id_token(pid2) # Psi_R^+
            
            elif self.helicity ==  Fraction(1,2) : # Psi_R
                # Psi_L^+ Psi_R
                mass_tokens_list += self.oppo_hand_partner.antiparticle.field_tokens + particle_id_token(pid1) # Psi_L^+
                mass_tokens_list += self.field_tokens                                + particle_id_token(pid2) # Psi_R
            else: raise ValueError("Helicity must be one of [+1/2,-1/2] for fermions")
                
        mass_tokens_list.append(CONTRACTIONS_TOKEN)
        mass_tokens_list.append(LORENTZ_TOKEN)
        mass_tokens_list += particle_id_token(pid1)
        mass_tokens_list += particle_id_token(pid2)
        if self.symmetry is not None:
            for sym in self.symmetry: 
                if sym.n == 1: continue
                mass_tokens_list += (sym.group_tokens) 
                mass_tokens_list += particle_id_token(pid1)  
                mass_tokens_list += particle_id_token(pid2)  # Need to double check how to contract in AUTOEFT convention [for antirep scenarios]

        # Second term: Psi_L^+ Psi_L^+ or Psi_R^+ Psi_R^+) or 
        mass_tokens_list.append("-")
        if majorana:
            # Psi_L^+ Psi_L^+ or Psi_R^+ Psi_R^+
            mass_tokens_list += self.antiparticle.field_tokens + particle_id_token(pid3) 
            mass_tokens_list += self.antiparticle.field_tokens + particle_id_token(pid4)
        else:
            if self.helicity ==  -Fraction(1,2) : # Psi_L
                # Psi_L^+ Psi_R
                mass_tokens_list += self.antiparticle.field_tokens         + particle_id_token(pid3 ) # Psi_L^+
                mass_tokens_list += self.oppo_hand_partner.field_tokens    + particle_id_token(pid4 ) # Psi_R  
            
            elif self.helicity ==  Fraction(1,2) : # Psi_R
                # Psi_L Psi_R^+
                mass_tokens_list += self.oppo_hand_partner.field_tokens    + particle_id_token(pid3) # Psi_L
                mass_tokens_list += self.antiparticle.field_tokens         + particle_id_token(pid4) # Psi_R^+  
            else: raise ValueError("Helicity must be one of [+1/2,-1/2] for fermions")
                

        mass_tokens_list.append(CONTRACTIONS_TOKEN)
        mass_tokens_list.append(LORENTZ_TOKEN)
        mass_tokens_list += particle_id_token(pid3)
        mass_tokens_list += particle_id_token(pid4)
        if self.symmetry is not None:
            for sym in self.symmetry: 
                if sym.n == 1: continue
                mass_tokens_list += (sym.group_tokens) 
                mass_tokens_list += particle_id_token(pid3)  
                mass_tokens_list += particle_id_token(pid4)  # Need to double check how to contract in AUTOEFT convention [for antirep scenarios]
        return mass_tokens_list
    
    
    


def display_term(term):
    term = term.copy()
    merged_list = [
        term[i] + term[i + 1] if term[i] == "-" and term[i + 1] == COMMUTATOR_TOKEN_A else term[i]
        for i in range(len(term) - 1)
        if not (i > 0 and term[i - 1] == "-" and term[i] == COMMUTATOR_TOKEN_A)
        ] + (term[-1:] if term[-2:] != ["-", COMMUTATOR_TOKEN_A] else [])
    term = merged_list
    contraction_mode= False
    for i in term:
        if i == "+" : contraction_mode = False
        if i == "-"+COMMUTATOR_TOKEN_A: 
            contraction_mode = False
            print("\n"+"-"+"\n"+COMMUTATOR_TOKEN_A,end=" ")
        elif (i == FIELD_TOKEN or i == spin_token(0)[0] or i == spin_token(1)[0] or i == spin_token(Fraction(1,2))[0] or i == DERIVATIVE_TOKEN or i == CONTRACTIONS_TOKEN  or i == GAMMA_TOKEN or i == SIGMA_BAR_TOKEN or i == SIGMA_TOKEN or i == "+" or i == "i" or i == COMMUTATOR_TOKEN_A or i == COMMUTATOR_TOKEN_B)          : 
            print("\n"+i,end=" ")
            if i == CONTRACTIONS_TOKEN: contraction_mode = True
        elif contraction_mode:
            if i == LORENTZ_TOKEN or i == "SU3" or i == "SU2" or i == "U1": 
                print("\n  "+i,end=" ")
            else: print(i, end=" ")
        else: print(i, end=" ")
         
        


# # Function 

# ## Get all symmetries and their groups


def get_all_symmetries(list_of_fields,tokens=False):
    """
    Get all the symmetry groups and symmetries of a list of fields.
    """
    all_groups = []
    all_symmetries = []
    for field in list_of_fields:
        if field.symmetry is not None:
            for sym in field.symmetry:
                if tokens:
                    if sym.group_tokens not in all_groups:
                        all_groups.append(sym.group_tokens)
                    if sym.tokens not in all_symmetries:
                        all_symmetries.append(sym.tokens)
                else:
                    if sym not in all_symmetries:
                        all_symmetries.append(sym)
                    if sym.group_name not in [ig.group_name for ig in all_groups]:
                        all_groups.append(sym)
    if tokens: 
        return all_groups, all_symmetries    
    else:
        return Symmetry.sort_groups(all_groups), Symmetry.sort_sym(all_symmetries)    

 

def get_group_dict(list_of_fields):
    """
    Get all the symmetry groups and symmetries of a list of fields.
    """

    all_groups = {}
    list_of_sym = []
    for field in list_of_fields:
        if field.symmetry is not None:
            for sym in field.symmetry:
                    list_of_sym.append(sym)
                    if sym.group_name not in all_groups.keys():
                        all_groups[sym.group_name] = {}
                        all_groups[sym.group_name]["TOK"]           = sym.group_tokens
                        all_groups[sym.group_name]["N"]             = sym.n
                        all_groups[sym.group_name]["LABEL"]         = sym.group_label
    all_groups["Lorentz"] = {}
    all_groups["Lorentz"]["TOK"] = [LORENTZ_TOKEN]
    
    #ordered_groups
    reordered_dict = OrderedDict((key, all_groups[key]) for key in ["Lorentz",*Symmetry.sort_groups(list_of_sym)])
    return reordered_dict

 
# CONTRACTION OF SYM IN THIS TERM?


def get_gauge_kinetic_terms(group_tokens):
    """
    Get the gauge kinetic terms given group tokens.
    """
    gauge_kinetic_terms = []
    ind1, ind2, ind3, ind4 = random.sample(INDICES_RANGE,4)

    # - 
    gauge_kinetic_terms.append("-")
    
    # [D_mu, D_nu]
    gauge_kinetic_terms.append(COMMUTATOR_TOKEN+"_A")
    gauge_kinetic_terms.append(DERIVATIVE_TOKEN)
    gauge_kinetic_terms += group_tokens
    gauge_kinetic_terms += derivative_id_token(ind1)

    gauge_kinetic_terms.append(COMMUTATOR_TOKEN+"_B")
    gauge_kinetic_terms.append(DERIVATIVE_TOKEN)
    gauge_kinetic_terms += group_tokens
    gauge_kinetic_terms += derivative_id_token(ind2)
    
    # [D^mu, D^nu]
    gauge_kinetic_terms.append(COMMUTATOR_TOKEN+"_A")
    gauge_kinetic_terms.append(DERIVATIVE_TOKEN)
    gauge_kinetic_terms += group_tokens
    gauge_kinetic_terms += derivative_id_token(ind3)

    gauge_kinetic_terms.append(COMMUTATOR_TOKEN+"_B")
    gauge_kinetic_terms.append(DERIVATIVE_TOKEN)
    gauge_kinetic_terms += group_tokens
    gauge_kinetic_terms += derivative_id_token(ind4)

    # CONTRACTIONS
    gauge_kinetic_terms.append(CONTRACTIONS_TOKEN)

    # Lorentz
    
    # D_mu and  D^mu
    gauge_kinetic_terms+= [LORENTZ_TOKEN,*derivative_id_token(ind1),*derivative_id_token(ind3)]
    gauge_kinetic_terms+= [LORENTZ_TOKEN,*derivative_id_token(ind2),*derivative_id_token(ind4)]
    
    # - [D_mu, D_nu] [D^mu, D^nu]
    return gauge_kinetic_terms
  

def generate_invariant_u1_for_3fields(numerator_upper_limit,denominator_upper_limit,nfields=3,verbose=False):
    """ 
    Generate a U1 for 3 fields.
    """
    # Generate the denominator
    denominator = random.randint(1,denominator_upper_limit)
    # Generate the fraction
    while True:
        x=Fraction(random.randint(0,numerator_upper_limit),denominator)
        y=Fraction(random.randint(0,numerator_upper_limit),denominator)

        if nfields == 2:
           z = random.choice([x,y])
        elif nfields == 3:
           z=Fraction(random.randint(0,numerator_upper_limit),denominator)
        strcomb = [(str(i)+str(x)+str(j)+str(y)+str(k)+str(z),(i,j,k))  for i,j,k in list(itertools.product(["+", "-"], repeat=3))]
        strcomb = [(i+" = "+str(eval(i)),sign)  for i,sign in strcomb if eval(i) == 0]
        
        if len(strcomb) > 0:
            possible_cases = [(i,(eval(sign[0]+str(1))*x,eval(sign[1]+str(1))*y,eval(sign[2]+str(1))*z)) for i,sign in strcomb]
            outstr,outqn   = random.choice(possible_cases)
            if verbose: print(outstr)
            assert sum(outqn) == 0, f"Sum of the quantum numbers should be zero. {outqn} != 0"
            if nfields == 2: 
                if not all([i==0 for i in outqn]):
                    assert outqn[0] != outqn[1], "The quantum numbers should be different"
                return outqn[0],outqn[1]
            return outqn


# ## Functions to check for sun rep for trilinear term and yukawa


def call_mathematica_code(mathematica_code):
    # Construct the command to call Mathematica with the code as input
    command = ['wolframscript', '-code', mathematica_code]
    
    try:
        # Execute the command
        output = subprocess.check_output(command)
        # Decode the output bytes to a string
        output = output.decode('utf-8')
        return output.strip()  # Remove leading/trailing whitespace
    except subprocess.CalledProcessError as e:
        # Handle any errors
        print("Error:", e)
        return None

def is_form(x,n=3):
    if isinstance(x, (tuple)) and len(x) == 2:
        x1,x2 = x
        if n !=2:
            if isinstance(x1, (tuple)) and isinstance(x2, (int)):
                return True
        else:
            if isinstance(x1, (int)) and isinstance(x2, (int)):
                return True

    return False

def check_sun_name(n,dynkin):
    """
    Check the name of the representation but also if the Dynkin numbers are a valid representation of SU(N).
    """
    math_code = f"<< GroupMath` \n"
    math_code += f"Print[RepName[SU{n}, "
    math_code += "{"
    for i in dynkin:
        math_code += str(i)+","
    math_code = math_code[:-1]
    math_code +="}]]\n"
    print("Mathematica code:")
    print("\t"+math_code.replace("\n","\n\t"))
    print("Running the code...")
    result = call_mathematica_code(math_code)
    
    print("Mathematica result:")
    print(result.split("\n")[-1])
    return result.split("\n")[-1]

def check_sun_inv(n,list_of_dynkin,verbose=False):
    """
    Check if the product of the representations of SU(N) is a singlet.
    """
    math_code = f"<< GroupMath` \n"
    math_code += "ReduceRepProduct[SU{}".format(n)
    math_code += ", {"
    for dynkin in list_of_dynkin:
        if dynkin is None:
            dynkin = tuple(np.zeros(n-1,dtype=int))
        
        math_code += "{"
        for i in dynkin:
            math_code += str(i)+","
        math_code = math_code[:-1]
        math_code += "},"

    math_code = math_code[:-1]
    math_code += "}, UseName -> False]"
    
    if verbose:
        print("Mathematica code:")
        print("\t"+math_code.replace("\n","\n\t"))
        print("Running the code...")
    
    result = call_mathematica_code(math_code)
    
    if verbose:
        print("Mathematica result:")
        for i in result.split("\n"):
            print("\t"+i)

    result = eval(result.split("\n")[-1].replace("{","(").replace("}",")"))

    if verbose:
        print("Used result:",result)
    
    if is_form(result,n):
        if (result[0]==tuple(np.zeros(n-1,dtype=int)) and n != 2) or (result[0]==0 and n == 2):
            if verbose: print("FOUND SINGLET!",list_of_dynkin,result,"\n")
            return True
    else:
        for isym in result:
            if is_form(isym,n):
                if (isym[0]==tuple(np.zeros(n-1,dtype=int)) and n != 2) or (isym[0]==0 and n == 2):
                    if verbose: print("FOUND SINGLET!",list_of_dynkin,isym,"\n")
                    return True
    if verbose:print(result)
    return False

 
# ### generate from predefined list for SUN invariant:


def generate_invariant_sun_for_3fields(n):
    if n == 2:
        x,y,z = random.sample(LIST_OF_SU2_INVARIAT_3FIELDS,1)[0]
    elif n == 3:
        x,y,z = random.sample(LIST_OF_SU3_INVARIAT_3FIELDS,1)[0]
    else:
        raise NotImplementedError("Only SU(2) and SU(3) are implemented")
    return x,y,z
 


def generate_trilinear_fields_qn(numerator_upper_limit=100,denominator_upper_limit=10,verbose=False):
    #@todo allow condition for su3 and su2
    x_su3 , y_su3 , z_su3 = generate_invariant_sun_for_3fields(3)
    x_su2 , y_su2 , z_su2 = generate_invariant_sun_for_3fields(2)
    x_u1  , y_u1  , z_u1  = generate_invariant_u1_for_3fields(numerator_upper_limit,denominator_upper_limit,3,verbose)
    x_sym = []
    y_sym = []
    z_sym = []
    try: 
        x_sym.append(Symmetry(n=1, rep=x_u1))
    except: pass
    try: 
        x_sym.append(Symmetry(n=2, rep=x_su2))
    except: pass
    try: 
        x_sym.append(Symmetry(n=3, rep=x_su3))
    except: pass
    try: 
        y_sym.append(Symmetry(n=1, rep=y_u1))
    except: pass
    try: 
        y_sym.append(Symmetry(n=2, rep=y_su2))
    except: pass
    try: 
        y_sym.append(Symmetry(n=3, rep=y_su3))
    except: pass
    try: 
        z_sym.append(Symmetry(n=1, rep=z_u1))
    except: pass
    try: 
        z_sym.append(Symmetry(n=2, rep=z_su2))
    except: pass
    try: 
        z_sym.append(Symmetry(n=3, rep=z_su3))
    except: pass

    return x_sym, y_sym, z_sym
  
def get_directories_in_path(path):
    # Get a list of all entries in the path
    entries = os.listdir(path)

    # Filter the entries to include only directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]

    return directories

# Function to extract "eps()" patterns from an input string
def extract_eps_patterns(input_string):
    # Define the pattern to match "eps()" patterns
    pattern = r'eps\((.*?)\)'
    
    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)
    
    return matches


def convert_autoeft_field_to_field(field):
    #print("Field    :",field)
    dagger = CONJUGATION_SYMBOL in str(field)
    #print("Dagger   :",dagger)
    sym_list = []
    for group_name,irep in field.representations.items():
        #group_name = re.findall(r'\((.*?)\)', k)
        #assert len(group_name) <= 1, "There can only be one group name: "+str(group_name)+","+str(k)
        #if group_name == [] : group_name = None
        #else                : group_name = group_name[0]
        #if 
        if group_name in ["SU3","SU2","U1"]:
            group_name = None

        if isinstance(irep,ae.base.representations.LorentzRepr):
            spin      = abs(irep.helicity)
            helicity  = irep.helicity
            if helicity == 0: helicity = None
            #print("\t",group_name,k,irep.helicity)
        
        elif isinstance(irep,ae.base.representations.SUNRepr):
            #print("\t",group_name,k,type(irep.partition),irep.partition == [],tuple(irep.partition) == tuple([]))
            if tuple(irep.partition) == tuple([]) : 
                continue
            if irep.N == 3:
                #print(irep.partition)
                rep_name =  [name for name,idict in SU3_DICT.items() if tuple(irep.partition) == tuple(idict["PARTITION"])]
                assert len(rep_name) == 1, "There should be only one name for the representation: "+str(rep_name)
                rep_name = rep_name[0]
            elif irep.N == 2:
                rep_name =  irep.partition[0]+1
            sym_list.append(Symmetry(n=irep.N, rep=rep_name,group_label=group_name))
        elif isinstance(irep,ae.base.representations.U1Repr):
            #print("\t",group_name,k,irep.charge)
            if irep.charge == 0 : continue
            sym_list.append(Symmetry(n=1, rep=irep.charge,group_label=group_name))
        else:
            raise ValueError("irep not a valid representation object from AutoEFT")
                        

    outfield = Field(spin        = spin, 
                     helicity    = helicity,
                     anti        = dagger,
                     symmetry    = sym_list,
                     generation  = field.generations, 
                     symbol        = str(field), 
                     tex_name    = field.tex, 
                     tex_hc_name = field.tex_hc, 
                     anticommute = field.anticommute)
    #print(outfield)
    #print(field.__repr__())
    #display(field.representations)
    return outfield

# # ## Class

# class Lagrangian:
#     def __init__(self,fields=None,lagrangian_string=None,autoeft_output_path=None,n_trilinear=None,name=None,description=None,run_name=None,dim=[1,2,3,4]):
#         # Can only be initilized either by input fields or by a lagrangian string or by an autoeft output path, not any combination of the three
#         assert (fields is not None) or (lagrangian_string is not None) or (autoeft_output_path is not None)  , "Should have either fields or lagrangian_string or autoeft_output_path"
#         assert not (fields is not None and lagrangian_string is not None)                                    , "Should not have both fields and lagrangian_string"
#         assert not (fields is not None and autoeft_output_path is not None)                                  , "Should not have both fields and autoeft_output_path"
#         assert not (lagrangian_string is not None and autoeft_output_path is not None)                       , "Should not have both lagrangian_string and autoeft_output_path"
        
#         # Check if the lagrangian string is valid
#         if lagrangian_string is not None:
#             assert isinstance(lagrangian_string,str), "Lagrangian string must be a string"

#         self.fields               = fields
#         self.lagrangian_string    = lagrangian_string
#         self.autoeft_output_path  = autoeft_output_path
#         self.input_yml_path       = None
#         self.outdir_path          = None

#         self.tokens         = []
#         self.name           = name
#         self.description    = description
#         self.run_name      = run_name
#         self.dim            = dim
#         self.n_trilinear    = n_trilinear
#         self.sym_group_dict = None
        

#         if self.fields is not None:
#             # Check if the fields are valid
#             for field in fields: assert isinstance(field,Field), "Fields must be a list of Field objects"
#             self.fields           = fields

#             if self.run_name is None: self.run_name = "test"

#         elif self.autoeft_output_path is not None:
#             # check if the directory exists
#             if not os.path.exists(self.autoeft_output_path) : raise FileNotFoundError("AutoEFT output path not found: "+self.autoeft_output_path)
#             if ".tar.gz" in self.autoeft_output_path:
#                 print("Extracting AutoEFT Output:",self.autoeft_output_path)
#                 try:
#                     output = subprocess.check_output(['tar', '-xzvf',self.autoeft_output_path])
#                     output = output.decode('utf-8')
                    
#                 except subprocess.CalledProcessError as e:
#                     print("Error:", e)
#                     raise e


#                 self.autoeft_output_path = self.autoeft_output_path.replace(".tar.gz","")

#             assert self.run_name is None, "Run name should not be provided when using AutoEFT output path"

#             self.input_yml_path        = os.path.join(self.autoeft_output_path,"model.yml")
#             self.outdir_path           = os.path.join(self.autoeft_output_path,"efts/model-eft/")

#             self.fields = self.read_fields_from_autoeft(dim_list=self.dim)
         
#         self.nscalar          = len([i for i in self.fields if i.spin == 0])
#         self.nvector          = len([i for i in self.fields if i.spin == 1])
#         self.nfermion         = len([i for i in self.fields if i.spin == Fraction(1,2)])
#         self.nfields          = len(self.fields)
#         self.n_singlets        = len([i for i in self.fields if i.symmetry == []])

#         self.symmetry_groups        , self.symmetry          = get_all_symmetries(self.fields,tokens=False)
#         self.symmetry_groups_tokens , self.symmetry_tokens   = get_all_symmetries(self.fields,tokens=True)
#         self.sym_group_dict                                  = get_group_dict(self.fields)

#         self.field_tokens = []
#         self.field_tokens_flat = []
#         for field in self.fields:
#             self.field_tokens.append(field.get_field_tokens())
#             self.field_tokens_flat += field.get_field_tokens()
        
#         self.field_strings = " ".join(self.field_tokens_flat)
        
#         # some sanity checks
#         assert  self.fields          is not None, "Fields should not be None"
#         assert len(self.symmetry_groups)+1 == len(self.sym_group_dict.keys())
#         assert self.nscalar+ self.nvector+ self.nfermion == self.nfields

#         # check for repeating fields
#         for i in self.fields: 
#             assert len([j for j in self.fields if i == j]) == 1 , "Fields should not repeat (Generation stuff not implemented yet). Repeating field: "+str(i)
#             assert i.anti == False                              , "Define Lagrangian by particles only"
#             assert CONJUGATION_SYMBOL not in i.symbol        , "Define Lagrangian by non-daggered objects only"

#     def __str__(self) -> str:
#          return str(self.nfields)+" : "+", ".join([i.__str__() for i in self.fields])
    
#     def __eq__(self, value: object) -> bool:
#         raise NotImplementedError("Not implemented yet")
    
#     def display(self):
#         print("Symmetry groups:",self.symmetry_groups)
#         print()
#         print("Kinetic terms:")
#         for term in self.kinetic_terms:
#             display_term(term)
#             print()
#         print("Mass terms:")
#         for term in self.mass_terms:
#             display_term(term)
#             print()
#         print("Interactions:")
#         for term in self.interaction_terms:
#             display_term(term)
#             print()
    
#     def get_fields(self):
#         return self.fields

#     def read_fields_from_autoeft(self,dim_list=[1,2,3,4]):
#         print("Reading AutoEFT Output:",self.outdir_path)
#         for idim in dim_list:
#             #print("Reading AutoEFT Output for dim:",idim)
#             dimpath        = os.path.join(self.outdir_path,str(idim),"basis")
#             if not os.path.exists(dimpath): print("Basis path not found: "+dimpath)

#             basis_file  = BasisFile(Path(dimpath))
#             model       = basis_file.get_model()
#             fields_dict = {}

#             for ik in model.fields.keys():
#                 if ik not in fields_dict: fields_dict[ik] = convert_autoeft_field_to_field(model.fields[ik])

#             if get_group_dict([fields_dict[ik] for ik in model.fields.keys()]) != self.sym_group_dict and self.sym_group_dict is not None:
#                 raise ValueError("Symmetry groups are not the same")

#             if [fields_dict[ik] for ik in model.fields.keys()] != self.fields and self.fields is not None:
#                 raise ValueError("Fields are not the same")
            
#             self.fields         = [fields_dict[ik] for ik in model.fields.keys()]
#             self.sym_group_dict = get_group_dict(self.fields)
        
#         self.fields           = [ifield for ifield in self.fields if CONJUGATION_SYMBOL not in ifield.symbol]
#         return self.fields

#     def get_kinetic_terms(self):
#         self.kinetic_terms      = []
#         self.kinetic_terms_flat = []
#         for field in self.fields:
#             self.kinetic_terms.append(field.get_kinetic_term())
#             self.kinetic_terms_flat += field.get_kinetic_term()
#         for group in self.symmetry_groups_tokens:
#             self.kinetic_terms.append(get_gauge_kinetic_terms(group))
#             self.kinetic_terms_flat += get_gauge_kinetic_terms(group)
#         return self.kinetic_terms
    
#     def check_partner(self,field):
#         for ifield in self.fields:
#             if ifield == field.get_oppo_hand_partner(): return True
#         return False
    
#     def make_mass_2fermions(self,field1,field2):
#         #print()
#         #print(field1.name,field2.name)
#         pid1, pid2 = random.sample(INDICES_RANGE,2)
        
        
#         mass_tokens_list = ["-"]
#         mass_tokens_list += field1.field_tokens + particle_id_token(pid1) # Psi_1
#         mass_tokens_list += field2.field_tokens + particle_id_token(pid2) # Psi_2
        
#         mass_tokens_list.append(CONTRACTIONS_TOKEN)
#         mass_tokens_list.append(LORENTZ_TOKEN)
#         mass_tokens_list += particle_id_token(pid1)
#         mass_tokens_list += particle_id_token(pid2)
#         if field1.symmetry is not None:
#             for sym in field1.symmetry: 
#                 if sym.n == 1: continue
#                 mass_tokens_list += (sym.group_tokens) 
#                 mass_tokens_list += particle_id_token(pid1)  
#                 mass_tokens_list += particle_id_token(pid2)  # Need to double check how to contract in AUTOEFT convention [for antirep scenarios]
#         return mass_tokens_list
    
#     def make_mass_2scalars(self,field1,field2):
#         raise KeyError("TODO: need update, bug exist, contractions when there is U1 only. Please fix it!")
#         pid1, pid2 = random.sample(INDICES_RANGE,2)

#         mass_tokens_list = ["+"]
        
#         # Phi^dagger Phi 
#         mass_tokens_list += field1.field_tokens + particle_id_token(pid1)
#         mass_tokens_list += field2.field_tokens + particle_id_token(pid2)

#         # U or SU Groups
#         if field1.symmetry is not None:
#             # CONTRACTIONS
#             mass_tokens_list.append(CONTRACTIONS_TOKEN)
#             for sym in field1.symmetry: 
#                 if sym.n == 1: continue
#                 mass_tokens_list+= (sym.group_tokens)
#                 mass_tokens_list += particle_id_token(pid1)
#                 mass_tokens_list += particle_id_token(pid2)
        
#         # + Phi^dagger Phi
#         return mass_tokens_list
     
#     def make_mass_matrix_fermions(self,field,field_list,field_list_anti):
#         # Check if the field is a fermion
#         assert field.spin == Fraction(1,2), "Field should be a fermion"
#         assert all([i.spin == Fraction(1,2) for i in field_list]), "Field list should be fermions"
#         assert all([i.spin == Fraction(1,2) for i in field_list_anti]), "Field list should be fermions"

#         # Find 
#         all_field      = [field]+field_list+[k.get_antiparticle() for k in field_list_anti]
#         all_field_anti = [field.get_antiparticle()]+field_list_anti+[k.get_antiparticle() for k in field_list]

#         lh = [k for k in all_field if k.helicity == Fraction(1,2)]
#         rh = [k for k in all_field if k.helicity == -Fraction(1,2)]

#         lh_anti = [k for k in all_field_anti if k.helicity == Fraction(1,2)]
#         rh_anti = [k for k in all_field_anti if k.helicity == -Fraction(1,2)]

#         # # all permutation pairigin
#         lh_iter_pairs = list(itertools.product(lh,lh_anti))
#         mass_term = [self.make_mass_2fermions(k,j) for k,j in lh_iter_pairs]

#         rh_iter_pairs = list(itertools.product(rh,rh_anti))
#         mass_term += [self.make_mass_2fermions(k,j) for k,j in rh_iter_pairs]
        
#         return mass_term
    
#     def make_mass_matrix_scalars(self,field,field_list,field_list_anti):
#         assert field.spin == 0, "Field should be a scalar"
#         assert all([i.spin == 0 for i in field_list]), "Field list should be scalars"
#         assert all([i.spin == 0 for i in field_list_anti]), "Field list should be scalars"

#         # Find 
#         all_field      = [field]+field_list+[k.get_antiparticle() for k in field_list_anti]
#         all_field_anti = [field.get_antiparticle()]+field_list_anti+[k.get_antiparticle() for k in field_list]

#         iter_pairs     = list(itertools.product(all_field,all_field_anti))
#         mass_term      = [self.make_mass_2scalars(k,j) for k,j in iter_pairs]

#         return mass_term

#     def make_mass_matrix_vectors(self,field_list):
#         raise NotImplementedError("Not implemented yet") 

#     def get_mass_terms(self):
#         self.mass_terms      = []
#         self.mass_terms_flat = []
#         made_mass_terms      = []
        
#         for field in self.fields:
#             if field in made_mass_terms: continue

#              # find field with same spin -> find same QN (indepdnet of dagger/anti/helicity
#             same_field      = [jfield for jfield in self.fields if field.check_eq_qn(jfield,False,False) and jfield is not field]
#             same_field_anti = [jfield for jfield in self.fields if field.check_eq_qn(jfield.get_antiparticle(),False,False) and jfield is not field]
           

#             if len(same_field )!=0 or len(same_field_anti)!=0: 
#                 for ifield in [field]+same_field+same_field_anti:
#                     made_mass_terms.append(ifield)

#                 if field.spin == Fraction(1,2): 
#                     for imass in self.make_mass_matrix_fermions(field,same_field,same_field_anti):
#                         self.mass_terms.append(imass)
#                         self.mass_terms_flat += imass

#                 elif field.spin == 0: 
#                     for imass in self.make_mass_matrix_scalars(field,same_field,same_field_anti):
#                         self.mass_terms.append(imass)
#                         self.mass_terms_flat += imass

#                 elif field.spin == 1: 
#                     raise NotImplementedError("Not implemented yet")
#                 else:
#                     raise ValueError("Spin not valid")

#             else:
#                 made_mass_terms.append(field)
#                 if field.spin == Fraction(1,2): 
#                     if field.symmetry == [] : 
#                         imass = field.get_mass_term(majorana=True)
#                         self.mass_terms.append(imass)
#                         self.mass_terms_flat += imass
                    
#                 elif field.spin == 0: 
#                     imass = field.get_mass_term()
#                     self.mass_terms.append(imass)
#                     self.mass_terms_flat += imass

#                 elif field.spin == 1: 
#                     raise NotImplementedError("Not implemented yet")
#                 else:
#                     raise ValueError("Spin not valid")
#         return self.mass_terms

#     def get_interactions(self,run_name=None,aemode="DYNKIN",run_mode=True,dim=None,tar=True):
#         if dim is None: dim = self.dim
#         if run_mode:
            
#             if self.autoeft_output_path is None:
#                 if run_name is not None: self.run_name = run_name
                
#                 self.autoeft_output_path        = os.path.join(BASE_AUTOEFT_OUTPUT_DIR,self.run_name)
#                 # check if the directory exists
#                 if not os.path.exists(self.autoeft_output_path):
#                     os.makedirs(self.autoeft_output_path)
    
#                 self.input_yml_path  = os.path.join(self.autoeft_output_path,"model.yml")
#                 self.outdir_path     = os.path.join(self.autoeft_output_path,"efts/model-eft/")

#             # Run the autoeft
#             if  os.path.exists(self.input_yml_path):
#                 os.remove(self.input_yml_path)
#             if  os.path.exists(self.outdir_path):
#                 shutil.rmtree(self.outdir_path)
            
#             self.write_autoeft_input(self.input_yml_path,aemode)
#             # Run the autoeft
#             os.chdir(self.autoeft_output_path)
#             print("Changed directory to:",os.getcwd())

#             # run it 
#             log_file = []
#             for idim in dim:
#                 print("Running AutoEFT for dim:",idim)
#                 # run autoeft and save log file
#                 log_file.append(self.run_autoeft_input(self.autoeft_output_path,self.input_yml_path,idim)            )

#             # merge log file
#             with open(os.path.join(self.autoeft_output_path,"log.txt"), 'w') as outfile:
#                 for fname in log_file:
#                     with open(fname,'r') as infile:
#                         outfile.write(infile.read())
#                     # remove the log file
#                     os.remove(fname)
#             os.remove("autoeft.log")

#         # Get the interactions
#         self.interaction_terms     = []
#         self.interaction_terms_flat = []
#         print("Reading AutoEFT Output:",self.outdir_path)
#         for idim in dim:
#             print("Reading AutoEFT Output for dim:",idim,end=" ---> ")
#             dimpath        = os.path.join(self.outdir_path,str(idim),"basis")
#             if not os.path.exists(dimpath): raise FileNotFoundError("Basis path not found: "+dimpath)
#             # If there is basis folder is empty, continue
#             basis_files = get_directories_in_path(dimpath)
#             if basis_files == []:
#                 print("\tBasis path is empty")
#                 continue
#             print("\tBasis path:",[os.listdir(os.path.join(dimpath,ibasis)) for ibasis in basis_files])

#             term = self.read_autoeft_output(dimpath)
#             self.interaction_terms += term.copy()

#             #print()
#         self.interaction_terms_flat = [j for i in self.interaction_terms for j in i]       

#         if tar: 
#             with tarfile.open(self.autoeft_output_path+".tar.gz", "w:gz") as tar: 
#                 arcname = os.path.relpath(self.autoeft_output_path, start=os.path.dirname(self.autoeft_output_path))
#                 tar.add(self.autoeft_output_path , arcname=arcname )

#                 # dont remove main files
#                 assert self.autoeft_output_path != BASE_AUTOEFT_OUTPUT_DIR, "Should not remove the base directory"
                
#                 n=1
#                 while True:
#                     path = BASE_AUTOEFT_OUTPUT_DIR.split("/")[:-n]
#                     parentfile = "/".join(path)
#                     if not os.path.exists(parentfile): break
#                     #print(parentfile,os.path.exists(parentfile))
#                     assert self.autoeft_output_path != parentfile, "Should not remove the base directory"
#                     n += 1
#                 shutil.rmtree(self.autoeft_output_path)
#         os.chdir(BASE_AUTOEFT_OUTPUT_DIR)
                
        
#         return self.interaction_terms, self.interaction_terms_flat
 
#     def write_autoeft_input(self,input_yml_path,aemode="DYNKIN"):
#         # assert number of fields is none zero
#         assert self.nfields != 0, "Number of fields is zero"

#         # Create the data dictionary for the yaml file
#         if self.name is None         : self.name = "BSM Model ({}S,{}V,{}F)".format([field.symbol[0] for field in self.fields].count("S"),[field.symbol[0] for field in self.fields].count("V"),[field.symbol[0] for field in self.fields].count("F"))
#         if self.description is None  : self.description =  "Model with : $"+", ".join([field.tex_name for field in self.fields]) +"$"

#         # Writing the input yaml file
#         input_yaml                = {}
#         input_yaml['name']        = self.name
#         input_yaml['description'] = self.description
#         input_yaml['symmetries']  = {"lorentz_group": {"tex": "SO^+(1,3)"}}
        
#         # Writing the Symmetry groups
        
#         # Writing the SU(N) groups
#         self.sun_sym = [isym for isym in self.symmetry if isym.n > 1]
#         if len(self.sun_sym) > 0:
#             input_yaml['symmetries']['sun_groups'] = {}
#             for isun_sym in self.sun_sym:
#                 if isun_sym.group_name not in input_yaml['symmetries']['sun_groups']:
#                     input_yaml['symmetries']['sun_groups'][isun_sym.group_name] = {"N": isun_sym.n, "tex": isun_sym.group_tex}
        
#         # Writing the U(1) groups
#         self.u1_sym = [isym for isym in self.symmetry if isym.n == 1]
#         if len(self.u1_sym) > 0:
#             input_yaml['symmetries']['u1_groups'] = {}
#             for iu1_sym in self.u1_sym:
#                 if iu1_sym.group_name not in input_yaml['symmetries']['u1_groups']:
#                     input_yaml['symmetries']['u1_groups'][iu1_sym.group_name] = {"tex": iu1_sym.group_tex}

#         # Writing the fields
#         input_yaml['fields'] = {}
#         for ifield in self.fields:

#             assert isinstance(ifield,Field) , "Fields must be a list of Field objects"
#             assert ifield.anti == False     , "Anti fields are not supported yet"

#             # Check if the field symbol is valid
#             fsymbol = ifield.symbol
#             if not ( re.match(r'^[a-zA-Z]', ifield.symbol) and  re.match(r'^[a-zA-Z0-9]+$', ifield.symbol) and  re.match(r'^[a-zA-Z0-9]+[a-zA-Z0-9\+]$', ifield.symbol) ):
#                 #print(r'^[a-zA-Z]',re.match(r'^[a-zA-Z]', ifield.symbol) )
#                 #print(r'^[a-zA-Z0-9]+$',re.match(r'^[a-zA-Z0-9]+$', ifield.symbol) )
#                 #print(r'^[a-zA-Z0-9]+[a-zA-Z0-9\+]$',re.match(r'^[a-zA-Z0-9]+[a-zA-Z0-9\+]$', ifield.symbol))
#                 fsymbol = re.sub(r'[^a-zA-Z\s]', "", ifield.symbol, flags=re.IGNORECASE)
#                 #print(list(ifield.symbol),"to",list(fsymbol))

#             # Check if the field symbol already exists, if so, add a prime
#             while fsymbol in input_yaml['fields']:
#                 fsymbol = fsymbol + "p"

#             input_yaml['fields'][fsymbol] = {} 
#             input_yaml['fields'][fsymbol]["representations"] = {}
#             if ifield.spin == 1 or ifield.spin == Fraction(1,2):
#                 input_yaml['fields'][fsymbol]["representations"]["Lorentz"] = str(Fraction(ifield.helicity))
            
#             for isym in ifield.symmetry:
#                 if isym.n >  1:
#                     if aemode.upper() =="DYNKIN":
#                         if "(" not in str(isym.dynkin):
#                             input_yaml['fields'][fsymbol]["representations"][isym.group_name] = "("+str(isym.dynkin)+")"
#                         else:
#                             input_yaml['fields'][fsymbol]["representations"][isym.group_name] = str(isym.dynkin)
#                     elif   aemode.upper() =="PARTITION":
#                         if "[" not in str(isym.partition):
#                             input_yaml['fields'][fsymbol]["representations"][isym.group_name] = "["+str(isym.partition)+"]"
#                         else:
#                             input_yaml['fields'][fsymbol]["representations"][isym.group_name] = isym.partition.copy()
#                 else:
#                     input_yaml['fields'][fsymbol]["representations"]["U1"] = str(isym.rep)
            
#             if ifield.generation is not None:
#                 input_yaml['fields'][fsymbol]["generations"] = ifield.generation
            
            
#             if ifield.tex_name is not None    : input_yaml['fields'][fsymbol]["tex"]    = ifield.tex_name
#             if ifield.tex_hc_name is not None : input_yaml['fields'][fsymbol]["tex_hc"] = ifield.tex_hc_name


#             input_yaml['fields'][fsymbol]["tex"]    = fsymbol.replace("p","'").replace("S",r"\phi").replace("FL",r"{\psi}_{L}").replace("FR",r"{\psi}_{R}") 

#         # Create the input file
#         with open(input_yml_path, "w") as yaml_file:
#             print("Saved at: "+input_yml_path)
#             # Write the header
#             yaml_file.write("# This AutoEFT model file was generated by the script\n")
#             yaml.dump(input_yaml, yaml_file, default_flow_style=False, sort_keys=False)

#         #interface to Autoeft Here: run and load
#         return 

#     def run_autoeft_input(self,dir_path,input_yml_path,dim):
#         try:
#             output = subprocess.check_output(['autoeft', 'c',input_yml_path, str(dim)])
#             output = output.decode('utf-8')
#             log_file = os.path.join(dir_path,"autoeft_dim"+str(dim)+".log")
#             with open(log_file, "w") as file:
#                 file.write(output)

#         except subprocess.CalledProcessError as e:
#             print("Error:", e)
#             raise e
#         return log_file

#     def read_autoeft_output(self,dimpath,verbose=False):
#         # Load Output
#         basis_file           = BasisFile(Path(dimpath))
#         model                = basis_file.get_model()
#         basis                = basis_file.get_basis()
#         all_term_tokens_list = []

#         # Iterate over types in the "basis" subdirectory ("type" in AUTOEFT's glossary)
        
#         for itype_key,itype_obj in basis.items(): 
#             existing_term = []
        
#             if verbose:
#                 print("Basis :    ",itype_key)
#                 print("Basis type:",itype_obj.type)
        
#             if "D" in itype_obj.type[0]:
#                 raise NotImplementedError("Terms with derivatives not implemented yet")
                
                
#             # Iterate over the terms 
#             for term in itype_obj.expanded():
        
#                 if verbose:
#                     print("\t term             :",term)
#                     print("\t term.symmetry    :",term.symmetry     )
#                     print("\t term.n_operators :",term.n_operators  )
#                     print("\t term.op_type     :",term.op_type      )
#                     print("\t term.operators   :",term.operators    )
#                     print("\t term.symmetry    :",term.symmetry     )
#                     print("\t term.term        :",term.term         )


#                 ########################################
#                 #################CHECKS#################
#                 ########################################
                
#                 # @TODO: Did not implement generation scenarios    
#                 if len(term.operators) > 1: 
#                     raise NotImplementedError("Only one operator per term is implemented. Seem to be multigeneration scenarios")
                
#                 # Sanity check to ensure fields in the terms are in the field dictionary
#                 if not (all([isymk in model.fields.keys() for isymk in term.symmetry.keys()])):
#                     raise ValueError("field not in the model.fields dictionary")
                

#                 #####################################
#                 ################INIT#################
#                 #####################################
                
#                 # Convert autoeft fields to aei fields
#                 fields_dict = {}
#                 for ik in term.symmetry.keys():
#                     fields_dict[ik] = convert_autoeft_field_to_field(model.fields[ik])
 
#                 #####################################
#                 ########### DO TERMS ################
#                 #####################################
                
#                 # Iterate over individual terms
#                 for iterm in str(term).split(" + "):
                    
#                     #------------#
#                     #---CHECKS---#
#                     #------------#
#                     assert (len(iterm.split(" * "))) == 3                               , "There should be only 3 elements" 
#                     coefficient, contraction_info, objects_info = iterm.split(" * ")
#                     assert type(eval(coefficient)) == int                               , f"Coefficient must be an integer:{coefficient}"
#                     assert ("eps" in contraction_info or contraction_info == "")        , f"Contraction info must have epsilons or empty:{contraction_info}"

#                     # Ignoring basis scenarios with permutation symettries, taking all term that can exist without repeating
#                     if iterm.split("*")[1:] in existing_term : continue
#                     else                                     : existing_term.append(iterm.split("*")[1:])

#                     #------------------#
#                     #---CONTRACTIONS---#
#                     #------------------#                    

#                     # Init things for contractions
#                     set_of_contracted_indices = extract_eps_patterns(iterm)
#                     obj_list                  = [x for x in objects_info.replace(" ","").split("*") if x.split("(")[0] in term.symmetry.keys()]
#                     pid_list                  = random.sample(INDICES_RANGE,len(obj_list))
                    
#                     # Checks
#                     assert set_of_contracted_indices == extract_eps_patterns(contraction_info), "The extracted indices should match between using full term or contraction info"
#                     assert len(np.unique(pid_list))  == len(pid_list)                         , "There should be no repeated indices"
                    
                    
#                     # Init things for contractions
#                     obj_dict    = {}
#                     term_tokens = ["+"]

#                     # Iterate over fields
#                     for iobj,ipid in zip(obj_list,pid_list):
#                         obj_dict[ipid]          = {}
#                         obj_dict[ipid]["OBJ"]   = iobj
#                         obj_dict[ipid]["FIELD"] = fields_dict[iobj.split("(")[0]].copy()
#                         indices                 = re.findall(r'\((.*?)\)', iobj)

#                         assert len(indices ) <= 1 and isinstance(indices,list), "There should be only one set of index or none"
#                         if len(indices ) == 1: 
#                             indices = re.split(';|,', indices[0])
#                         obj_dict[ipid]["INDICES"] = indices

#                         # Save field tokens
#                         term_tokens += obj_dict[ipid]["FIELD"].field_tokens
#                         term_tokens += particle_id_token(ipid)
                    
#                     if verbose:
#                         print("\t\t iterm           :",iterm)
#                         print("\t\t\t Extracted Coefficient :",coefficient)
#                         print("\t\t\t Extracted indices     :", set_of_contracted_indices)
#                         print("\t\t\t Extracted pid_list    :", pid_list)
                    
#                         print("\t\t iterm           :",iterm)
#                         print("\t\t\t Coefficient       :", coefficient        )
#                         print("\t\t\t Extracted indices :", contraction_info   )
#                         print("\t\t\t pid_list          :", objects_info       )
#                         for k,i in obj_dict.items():
#                             print("\t\t\t\t ",particle_id_token(k),i["OBJ"],i["INDICES"])
                    
#                     # Do contractions
#                     contractions_tokens = []
                    
#                     # Iterate over symmetry group
#                     for isym in self.sym_group_dict.keys():
                        
#                         # find sets of contracted indices
#                         isym_set_of_contracted_indices = [ cind for cind in set_of_contracted_indices if isym in cind ]
#                         if isym_set_of_contracted_indices == []: continue

#                         # iterate over contracted indices
#                         for contracted_indices in isym_set_of_contracted_indices:
#                             eps_indices         = contracted_indices.split(",")
#                             contracting_fields = []

#                             # find fields according to their indices
#                             for indices in eps_indices:
#                                 contracting_fields  += [pid for pid,idict in obj_dict.items() if indices in idict["INDICES"]]

#                             assert contracting_fields is not [], "There should be contracting fields"

#                             # get contracted tokens
#                             conttok = self.sym_group_dict[isym]["TOK"].copy()
#                             for pidtok in [particle_id_token(i) for i in contracting_fields]:
#                                 conttok += pidtok.copy()
                            
#                             if verbose: print("\t\t\t\t\t ",conttok)
#                             contractions_tokens += conttok.copy()

#                     # Save term tokens        
#                     if contractions_tokens != []:   
#                         term_tokens += [CONTRACTIONS_TOKEN]
#                         term_tokens += contractions_tokens

#                     if verbose:
#                         display_term(term_tokens)
#                         print()
                    
#                     # Save term tokens
#                     all_term_tokens_list.append(term_tokens)
#         return all_term_tokens_list
    
#     def generate_tokens(self,inter="run",run_name=None):    
#         if run_name is None: run_name = self.run_name
        
#         self.get_kinetic_terms() 
#         self.get_mass_terms()
#         if isinstance(inter,str):
#             if inter.lower()  == "run": 
#                 self.get_interactions(run_name=run_name,run_mode=True)
#                 self.tokens =  self.interaction_terms_flat+ self.kinetic_terms_flat + self.mass_terms_flat 
#             elif inter.lower()  == "read": 
#                 assert (run_name is not None or self.autoeft_output_path is not None) , "Run name or autoeft_output_path should be provided"
#                 self.get_interactions(run_name=run_name,run_mode=False)
#                 self.tokens =  self.interaction_terms_flat+ self.kinetic_terms_flat + self.mass_terms_flat 
#             else:   
#                 raise ValueError("Interactions should be either 'run' or 'read' or None")
#         elif inter is None:
#             self.tokens = self.kinetic_terms_flat + self.mass_terms_flat
#         else:
#             raise ValueError("Interactions should be either 'run' or 'read' or None")
#         self.ntokens = len(self.tokens)
#         self.lagrangian_string = " ".join(self.tokens)

#         assert  isinstance(self.tokens,list)                 , "Tokens should be a list"
#         assert  all([isinstance(i,str) for i in self.tokens]), "Tokens should be a list of strings"
#         self.count_for_trilinear()
        
#         return self.tokens
    
#     def check_for_scalar_trilinear(self):
#         for iterm in self.interaction_terms: 
#             if " ".join(iterm).count("SPIN 0") == 3: return True
#         return False
    
#     def check_for_yukawa(self):
#         for iterm in self.interaction_terms: 
#             if " ".join(iterm).count("SPIN 1/2") == 2 and " ".join(iterm).count("SPIN 0") == 1: return True
#         return False

#     def count_for_scalar_trilinear(self):
#         n_scalar_trilinear = 0
#         for iterm in self.interaction_terms: 
#             if " ".join(iterm).count("SPIN 0") == 3: 
#                 n_scalar_trilinear += 1
#         return n_scalar_trilinear
    
#     def count_for_yukawa(self):
#         nyukawa = 0
#         for iterm in self.interaction_terms: 
#             if " ".join(iterm).count("SPIN 1/2") == 2 and " ".join(iterm).count("SPIN 0") == 1: 
#                 nyukawa += 1
#         return nyukawa
    
#     def count_for_trilinear(self):
#         self.n_scalar_trilinear = self.count_for_scalar_trilinear()
#         self.n_yukawa            = self.count_for_yukawa()
#         self.n_trilinear        = self.n_scalar_trilinear+ self.n_yukawa
#         return self.n_trilinear


#     def sanity_checks(self):
#         # Check if the lagrangian string is valid
#         assert isinstance(self.lagrangian_string,str)                 , "Lagrangian string must be a string"
#         assert self.ntokens == len(self.tokens)                       , "Number of tokens should match the number of tokens"
#         assert self.ntokens == len(self.lagrangian_string.split(" ")) , "Number of tokens should match the number of tokens"

#         # Check if the fields are valid
#         for field in self.fields: assert isinstance(field,Field), "Fields must be a list of Field objects"
#         assert isinstance(self.field_strings,str)                    , "Field strings must be a string"


#         print("Checking number of tokens")
#         print("Checking number of tokens in mass terms:",end="")
#         ntok_mass = 0
#         for imass in self.mass_terms: ntok_mass+=len(imass)
#         assert ntok_mass == len(self.mass_terms_flat), ntok_mass    
#         print(ntok_mass)

#         print("Checking number of tokens in kinetic terms:",end="")
#         ntok_kin = 0
#         for ikin in self.kinetic_terms: ntok_kin+=len(ikin)
#         assert ntok_kin  == len(self.kinetic_terms_flat), ntok_kin
#         print(ntok_kin)

#         print("Checking number of tokens in interaction terms:",end="")
#         ntok_int = 0
#         for iint in self.interaction_terms: ntok_int+=len(iint)
#         assert ntok_int  == len(self.interaction_terms_flat), ntok_int
#         print(ntok_int)

#         print("Checking number of tokens in all terms:",end="")
#         assert self.ntokens == ntok_mass + ntok_kin + ntok_int, self.ntokens
#         for i in self.tokens: assert i in self.lagrangian_string.split(" "), i
#         print(self.ntokens)

#         print("Checking number of field tokens:",end="")
#         nfield_tokens = 0
#         for ifield in self.fields: nfield_tokens += len(ifield.field_tokens)
#         assert nfield_tokens == len(self.field_tokens_flat), nfield_tokens
#         assert self.nfields == len(self.fields), self.nfields
#         assert self.nfields == len(self.field_tokens), self.nfields
#         print(nfield_tokens)
        
#         print("All checks passed")      
#         return True
    
    