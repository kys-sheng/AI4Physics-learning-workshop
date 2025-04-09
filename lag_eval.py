

import os
import sys
import numpy as np
import time
import json
import pickle
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# WordLevel tokenizer to serve as input to the Hugginface tokenizer
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration, BartConfig
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from transformers import BartTokenizer, BartModel

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection

from matplotlib.colors import LogNorm

from collections import defaultdict

from tqdm import tqdm
tqdm.pandas()

import ipywidgets as widgets
from IPython.display import display

import itertools

from fractions import Fraction
import aei_class as aei
import re
import random

from collections import Counter

#####################
##### Variables #####
#####################

all_id = [aei.DERIVATIVE_ID_TOKEN+str(i)  for i in  aei.INDICES_RANGE]+[aei.GAMMA_ID_TOKEN+str(i)  for i in  aei.INDICES_RANGE]+[aei.PARTICLE_ID_TOKEN+str(i)  for i in  aei.INDICES_RANGE]+[aei.SIGMA_ID_TOKEN+str(i)  for i in  aei.INDICES_RANGE]
all_id = list(np.unique(all_id)    )
vocab  = ["i","+","-","/","COMMUTATOR_A","COMMUTATOR_B","0","1","2","3","4","5","6","7","8","9","ID0","ID1","ID2","ID3","ID4","ID5","ID6","ID7","ID8","ID9",aei.FIELD_TOKEN,"SPIN","HEL","DAGGER",aei.DERIVATIVE_TOKEN,aei.SIGMA_BAR_TOKEN,"SU3","SU2","U1","LORENTZ",aei.CONTRACTIONS_TOKEN]

math_vocab          = ["i","+","-","/","COMMUTATOR_A","COMMUTATOR_B"]
int_vocab           = ["0","1","2","3","4","5","6","7","8","9"]
id_vocab            = all_id
in_sym_group_vocab  = ["SU3","SU2","U1"]
sym_group_vocab     = in_sym_group_vocab + ["LORENTZ"]
field_vocab         = [aei.FIELD_TOKEN,"SPIN","HEL","DAGGER",aei.DERIVATIVE_TOKEN,aei.SIGMA_BAR_TOKEN]
derivative_vocab    = [ aei.DERIVATIVE_TOKEN   ] + in_sym_group_vocab + list(all_id)
qn_vocab            = [ aei.CONTRACTIONS_TOKEN ] + sym_group_vocab    

rep_tex_dict = {
    "SU3":{"-3":r"\bar{\textbf{3}}","3":r"\textbf{3}"},
    "SU2":{"-2":r"\textbf{2}","2":r"\textbf{2}","-3":r"\textbf{3}","3":r"\textbf{3}"},
}

###########################
##### Term Separation #####
###########################
def sep_terms(instr,eos="[EOS]",sos="[SOS]",pad="[PAD]",ntok_error_show=10,completeness_check=False):
    
    initial_string = instr
    assert type(instr) == str, f"sep_terms function only take in lagrangian string: {type(instr)} found instead"
    
    # Remove special tokens
    instr = instr.replace(eos,"").replace(sos,"").replace(pad,"").replace("  "," ")
    while "  " in instr: instr = instr.replace("  "," ")
    
    # Make them into a list of tokens
    instr        = instr.split()
    sep_term_ind = []

    # Get index of tokens
    for i in np.arange(0,len(instr),1):
        try:
            if i+1 < len(instr):
                # (+/-)  (FIELD ... / D_mu ... / [ ... ) ... 
                if ((instr[i] == "+" or instr[i] == "-")  
                   and (instr[i+1] == aei.FIELD_TOKEN 
                        or instr[i+1] == aei.DERIVATIVE_TOKEN 
                        or instr[i+1] == aei.COMMUTATOR_TOKEN_A
                        or instr[i+1] == "i")  ):
                    sep_term_ind.append(i)
            if i+1 == len(instr):
                if (instr[i] == "+" or instr[i] == "-"): sep_term_ind.append(i)

            #if i+2 < len(instr):
            #    # + i FIELD .... 
            #    if ((instr[i] == "+" or instr[i] == "-")  
            #       and (instr[i+1] == "i")  
            #       and (instr[i+2] == aei.FIELD_TOKEN)):
            #       sep_term_ind.append(i)

        except Exception as e:
            raise KeyError(f"ERROR: {str(e)}")

    # Get list of term_list by index
    split_lists = [instr[i:j] for i, j in zip([0]+sep_term_ind, sep_term_ind+[None]) if instr[i:j] != []]
    
    if completeness_check:
        complete_flag = True
        for term_list in split_lists:
            if term_list[-1] not in all_id and aei.CONTRACTIONS_TOKEN != term_list[-1]:
                complete_flag = False
                break 
        return split_lists, complete_flag
    return split_lists

def get_tok_count_dict(instr):
    tok = aei.COMMUTATOR_TOKEN , aei.COMMUTATOR_TOKEN_A , aei.COMMUTATOR_TOKEN_B , aei.CONTRACTIONS_TOKEN , aei.DAGGER_TOKEN , aei.DERIVATIVE_ID_TOKEN , aei.DERIVATIVE_TOKEN , aei.FIELD_TOKEN , aei.GAMMA_ID_TOKEN , aei.GAMMA_TOKEN , aei.HELICITY_TOKEN , aei.LORENTZ_TOKEN , aei.PARTICLE_ID_TOKEN , aei.SIGMA_BAR_TOKEN , aei.SIGMA_ID_TOKEN , aei.SIGMA_TOKEN , aei.SPIN_TOKEN 
    tok_dat = {}
    for itok in tok:
        tok_dat[itok] = instr.count(itok)
    return tok_dat

#############################
##### Object Separation #####
#############################

def sep_objects(inlist,checks=False,verbose=False,group_tokens=["SU3", "SU2", "LORENTZ"]):
    # inlist is the list of tokens of a term 
    
    # type checks
    if type(inlist) == str:
        assert inlist not in vocab, f"Attempting to seperate a single token into terms! Double check! -> {inlist}"
        inlist = inlist.replace("  "," ").replace("  "," ").replace("  "," ").split()
    assert type(inlist) == list
    
    sep_term_ind         = []
    poly_token_object    = False
    contractions_object  = False 
    completeness         = True     
    good_contraction     = True         

    # index searching    
    for i in np.arange(0,len(inlist),1):
    
        # searching mode
        if poly_token_object == False and contractions_object == False:
            # polytoken object scenario
            if (inlist[i] == aei.FIELD_TOKEN
                or inlist[i] == aei.DERIVATIVE_TOKEN  
                or inlist[i] == aei.SIGMA_BAR_TOKEN
                or inlist[i] == aei.SIGMA_TOKEN
                or inlist[i] == aei.DERIVATIVE_TOKEN ): 
                
                sep_term_ind.append(i)
                poly_token_object = True

            # single token object scenario
            elif (inlist[i] == "+"
                or inlist[i] == "-"
                or inlist[i] == "i"
                or inlist[i] == aei.COMMUTATOR_TOKEN_A 
                or inlist[i] == aei.COMMUTATOR_TOKEN_B): 
                sep_term_ind.append(i)
           
            # contraction scenario
            elif (inlist[i] == aei.CONTRACTIONS_TOKEN) : 
                sep_term_ind.append(i)
                contractions_object = True
            
            # odd case, raise Error
            else:
                assert poly_token_object == False and contractions_object== False
                if verbose: print(f"'{inlist[i]}' is not a single token object")
                completeness     = False         
                good_contraction = False       
                break
            
        # polytoken mode [stopping check]
        elif poly_token_object == True:
            if inlist[i] in all_id :
                poly_token_object = False

        # contraction mode [sanity checks]
        elif contractions_object == True :
            if inlist[i] in [aei.FIELD_TOKEN,aei.DERIVATIVE_TOKEN, aei.SIGMA_BAR_TOKEN, aei.SIGMA_TOKEN, aei.DERIVATIVE_TOKEN, aei.COMMUTATOR_TOKEN_A, aei.COMMUTATOR_TOKEN_B] : 
                print(f"FOUND {inlist[i]} in Term ({inlist})")
                completeness     = False   
                good_contraction = False
                break

        else:
            if verbose:
                print( " poly_token_object              :",poly_token_object )
                print( " contractions_object            :",contractions_object )
                print(f" Current Token                  : {inlist[i]}")
                print(f" Rest of the Strings            : {inlist[i:]}")
                print(f" Last token of full lagrangian  : {inlist[-1]}")
            completeness     = False         
            good_contraction = False             
            break
    
    # split list accodrding to the indices        
    split_lists = [inlist[i:j] for i, j in zip([0]+sep_term_ind, sep_term_ind+[None]) if inlist[i:j] != []]
    assert " ".join([" ".join(i) for i in split_lists]).replace("  "," ").replace("  "," ") == " ".join(inlist).replace("  "," ").replace("  "," ")

    # Do checks
    if checks:
        instr = " ".join(inlist)
        
        # @TODO: This would be true in future
        #if aei.CONTRACTIONS_TOKEN not in inlist:
        #    good_contraction = True
        
        # @TODO: MODIFY FOR FUTURE (should be no contraction token if no contraction needed)
        
        # SYM GROUP CHECKS
        if (any([igroup   in instr for igroup in group_tokens]) or 
            any([ilorentz in instr for ilorentz in ["SPIN 1 / 2", aei.SIGMA_BAR_TOKEN, aei.SIGMA_TOKEN, aei.DERIVATIVE_TOKEN]])):  
            if not contractions_object:
                good_contraction = False
                if verbose:
                    for igroup in group_tokens:
                        if igroup in instr: print(f"'{igroup}' in term '{instr}' but no contraction part found")            
                    for ilorentz in ["SPIN 1 / 2", aei.SIGMA_BAR_TOKEN, aei.SIGMA_TOKEN, aei.DERIVATIVE_TOKEN]:
                        if ilorentz in instr: print(f"'{ilorentz}' in term '{instr}' but no contraction part found")            
        
        id_in_contract = []
        if contractions_object: 
            id_in_contract =  list(np.unique([tok for tok in split_lists[-1] if tok in all_id]))

        for obj in split_lists:
            # single token objects
            if (obj[0] == "+" or obj[0] == "-" or obj[0] == "i" or obj[0] == aei.COMMUTATOR_TOKEN_A or obj[0] == aei.COMMUTATOR_TOKEN_B) and len(obj) == 1:
                continue
            if contractions_object and len(id_in_contract) != 0:
                obj_id = [itok for itok in obj if itok in all_id]
                if len(obj_id) > 1 : completeness = False
                if len(obj_id) == 1 and obj_id[0] in id_in_contract: id_in_contract.remove(obj_id[0])

            # polytoken objects and contractions part 
            if obj[-1] not in all_id and obj[-1] != aei.CONTRACTIONS_TOKEN: # @TODO: MODIFY FOR FUTURE (should be no contraction token if no contraction needed)
                completeness = False
                if verbose: print(f"Polytoken objects '{obj}' did not end with ID" )
                #break 
        if contractions_object and len(id_in_contract) != 0: 
            good_contraction = False

        if verbose and (completeness == False or good_contraction == False):
            print("Broken Term:",instr)
        #print(split_lists,completeness,good_contraction)
        return split_lists,completeness,good_contraction
    return split_lists

def check_same_objects(term1, term2):
    assert type(term1) == str 
    assert type(term2) == str 

    term1_objects = sep_objects(term1)
    # remove IDs
    term1_objects = [[itok for itok in obj if itok not in all_id] for obj in term1_objects  ]
    # sort objects
    term1_objects.sort()
    
    term2_objects = sep_objects(term2)
    # remove IDs
    term2_objects = [[itok for itok in obj if itok not in all_id] for obj in term2_objects  ]
    # sort objects
    term2_objects.sort()
    
    # check if objects are the same
    if term1_objects == term2_objects:
        return True
    else:
        return False

def sanity_check_for_sep_functions(instr,eos="[EOS]",sos="[SOS]",pad="[PAD]"):
    terms = sep_terms(instr,eos="[EOS]",sos="[SOS]",pad="[PAD]")
    out_obj = []
    for i in terms:
        if not check_same_objects(" ".join(i)," ".join(i)):
            raise KeyError("check_same_objects function have problems")
        if not same_objects_Q(" ".join(i)," ".join(i)):
            raise KeyError("same_objects_Q function have problems")
        
        out_obj += sep_objects(i,verbose=False)
    
    instr = instr.replace(eos,"").replace(sos,"").replace(pad,"").replace("  "," ")
    while " " in instr[0]:
        instr = instr[1:]
    while " " in instr[-1]:
        instr = instr[:-1]
    while "  " in instr:
        instr = instr.replace("  "," ")
    
    outstr = " ".join([" ".join(i) for i in out_obj])
    while " " in outstr[0]:
        outstr = outstr[1:]
    while " " in outstr[-1]:
        outstr = outstr[:-1]
    while "  " in outstr:
        outstr = outstr.replace("  "," ")
    try:
        assert instr == outstr
    except:
        print("I:",instr,list(instr) )
        print("O:",outstr,list(outstr))
    return instr == " ".join([" ".join(i) for i in out_obj]).replace("  "," ")

#######################
##### CONTRACTION #####
#######################
def get_contraction_dict(contract_list,group_tokens):
    # Translate contraction(list of tokens) into a dictionary 
    assert type(contract_list) == list

    contract_dict = {}
    tok_ids       = []
    
    # Go through token by token
    for itok in contract_list:
        if itok == aei.CONTRACTIONS_TOKEN : continue

        #if sym group found
        if itok in group_tokens:

            # if accumulated tok_ids not empty, found another group token. Save and start again
            if tok_ids != []:
                contract_dict[last_sym].append(tok_ids)
                tok_ids    = []
               
            #if new sym group, start a dictionary for it
            if itok not in contract_dict:
               contract_dict[itok] = []
               tok_ids             = []
               last_sym            = itok

            #if old sym group, start a new list for it
            elif itok in contract_dict:
               tok_ids             = []
               last_sym            = itok
        else: 
            # Add tokens into a list 
            tok_ids.append(itok)
    
    # Save last case
    if tok_ids != []:
        contract_dict[last_sym].append(tok_ids)
        tok_ids    = []
    
    return  contract_dict

def get_contractions(term, group_tokens=["SU3", "SU2", "LORENTZ"]):
    assert aei.CONTRACTIONS_TOKEN in term, "No contraction in term"
    
    # Separate objects
    objs,completeness,good_contraction = sep_objects(term,checks=True)

    if not good_contraction:
        return None,None
    
    # Get contraction part
    contractions = [i for i in objs if any(j in all_id for j in i) and aei.CONTRACTIONS_TOKEN in i]
    assert len(contractions) == 1, f"There should be exactly one contraction part in a term.{len(contractions)} found: {term}"
    
    contractions = contractions[0]
    assert contractions.count(aei.CONTRACTIONS_TOKEN) == 1, f"Contraction token should appear exactly once. {contractions.count(aei.CONTRACTIONS_TOKEN)} found"
    
    # Get objects part (excluding the contraction part)
    objs = [i for i in objs if any(j in all_id for j in i) and aei.CONTRACTIONS_TOKEN not in i]
    
    # Get contraction dictionary
    contract_dict = get_contraction_dict(contractions, group_tokens)
    
    obj_dict = {}
    out_dict_pos = {}
    out_dict_cont = {}

    # Create object dictionary with positions and object details
    for n, obj in enumerate(objs):
        obj_id = [j for j in obj if j in all_id]
        assert len(obj_id) == 1, f"Each object must have exactly one identifier: {term}"
        obj_id = obj_id[0]
        
        obj_dict[obj_id] = {
            "pos": n,
            "obj": [j for j in obj if j not in all_id]
        }
    
    # Populate position and contraction dictionaries
    for group, id_sets in contract_dict.items():
        out_dict_pos[group] = []
        out_dict_cont[group] = []
        
        for id_set in id_sets:
            positions = [obj_dict[iid]["pos"] for iid in id_set]
            objects =   [obj_dict[iid]["obj"] for iid in id_set]
            
            # Ensure that all contractions are consistent with group tokens
            if group != "LORENTZ":
                if not all(group in obj_dict[iid]["obj"] for iid in id_set):
                    print(f"GROUP {group} mismatch in contraction: {term}")
                #assert all(group in obj_dict[iid]["obj"] for iid in id_set), f"GROUP {group} mismatch in contraction: {term}"
            
            out_dict_pos[group].append(positions)
            out_dict_cont[group].append(objects)
    
    # Sort positions and contractions to ignore order but maintain reoccurrence
    for group in out_dict_pos   : out_dict_pos[group] = sorted(sorted(inner) for inner in out_dict_pos[group])
    for group in out_dict_cont  : out_dict_cont[group] = sorted(sorted(inner) for inner in out_dict_cont[group])
    
    return out_dict_pos, out_dict_cont

def check_same_contractions(term1,term2,check_pos=False,group_tokens=["SU3", "SU2", "LORENTZ"]):    
   
   # If contractions only exist one side, then its definitely not the same
   if   aei.CONTRACTIONS_TOKEN not in term1 and aei.CONTRACTIONS_TOKEN in term2 : return False 
   elif aei.CONTRACTIONS_TOKEN not in term2 and aei.CONTRACTIONS_TOKEN in term1 : return False  
   
   # If contractions exist both side, then start checks
   elif aei.CONTRACTIONS_TOKEN in term1 and aei.CONTRACTIONS_TOKEN in term2 :
      
      # TODO: MOdify this in future, should be no contraction if no need for contraction
      # If contraction tokens is not the last tokens on both side
      if term1.split()[-1] != aei.CONTRACTIONS_TOKEN and term2.split()[-1] != aei.CONTRACTIONS_TOKEN : 
         out_dict_pos1, out_dict_cont1 = get_contractions(term1, group_tokens)
         out_dict_pos2, out_dict_cont2 = get_contractions(term2, group_tokens)
         
         if check_pos : return (out_dict_pos1 == out_dict_pos2 and out_dict_cont2 == out_dict_cont1)
         else         : return (out_dict_cont2 == out_dict_cont1)
      
      else:
         # If contraction tokens is not the last tokens on both side
         if   term1.split()[-1] != aei.CONTRACTIONS_TOKEN and term2.split()[-1] == aei.CONTRACTIONS_TOKEN : return False
         elif term1.split()[-1] == aei.CONTRACTIONS_TOKEN and term2.split()[-1] != aei.CONTRACTIONS_TOKEN : return False
         elif term1.split()[-1] == aei.CONTRACTIONS_TOKEN and term2.split()[-1] == aei.CONTRACTIONS_TOKEN : return True
         else: raise KeyError

   # If contractions only dont exist on both side, then its definitely Same
   else:
      return True

#################
##### SCORE #####
#################
def replace_ids(instr):
    newstr = instr
    uniq_tok = np.unique([tok for tok in instr.split() if tok in all_id])
    if len(uniq_tok) == 0: return instr
    new_tok  = random.sample([tok for tok in all_id if tok not in uniq_tok],4)
    for oldtok,newtok in zip(uniq_tok,new_tok):
        newstr = newstr.replace(oldtok,newtok)
    if newstr == instr: newstr = replace_ids(newstr)
    return newstr

def check_ids(instr):
    uniq_tok = np.unique([tok for tok in instr.split() if tok in all_id])
    if len(uniq_tok) == 0: return True
    else                 : return False

def get_lagrangian_score_ys(predicted, expected,verbose=False,force=False):
    if not force: 
        raise KeyError("Dont use this (we modified the score to have length penalty thtat takes in only extra tersm). Use get_lagrangian_score. If really want to use it, switch force=True")

    # 1. Separate terms
    # 2. Check for repeating terms, if repeat mod ID accordingly
    # 3. Get all possible combinations of terms based on number of tokens
    # 4. From those combinations get cases with same objects
    # 5. Check contractions if make sense
    # 6. Check all cases if there are true terms that was paired with more than 1 pred_term
    # 7.    If yes, check pred term is used elsewhere or not with guranteed one term match
    #          If yes, check pred term is used elsewhere or not with guranteed one term match
    #                If yes, delete in many term case
    #                If not, raise Error and modify code is needed
    #          If not,  take one of the prediction
    # 8 Calculate Score 

    assert sanity_check_for_sep_functions(expected) == True
    
    try                     : sanity_check_for_sep_functions(predicted)
    except Exception as e   : print(str(e))
    
    # 1. 
    # Seperate lagrangians to terms
    sep_term_true       , complete_true      = sep_terms(expected,completeness_check=True)
    sep_term_predicted  , complete_predicted = sep_terms(predicted,completeness_check=True)
    sep_term_true_joined                     = [" ".join(i) for i in sep_term_true  ]
    sep_term_predicted_joined                = [" ".join(i) for i in sep_term_predicted  ]

    # check for repeating terms
    assert len(sep_term_true_joined) == len(np.unique(sep_term_true_joined))

    # 2.
    # check for repeating terms in prediction, if so, change to new set of id just in case its a coincidence
    if len(sep_term_predicted_joined) != len(np.unique(sep_term_predicted_joined)):
        uniq_sep = []
        for term,nterm in dict(Counter(sep_term_predicted_joined)).items():
            if nterm != 1:
                for ind in range(nterm): 
                    new_term = replace_ids(term)
                    if len(np.unique([tok for tok in new_term.split() if tok in all_id])) !=0:
                        while (new_term in uniq_sep): new_term = replace_ids(term)
                    uniq_sep.append(new_term)
            else : 
                uniq_sep.append(term)
        sep_term_predicted_joined = uniq_sep
        sep_term_predicted        = [t.split() for t in  uniq_sep]
    
    # if there are terms that dont have ids and they repeat, then its fine. Otherwise check
    if all([check_ids(instr) for instr in sep_term_predicted_joined]):
        assert len(sep_term_predicted_joined) == len(np.unique(sep_term_predicted_joined))

    # 3.
    # Get all combination of term_true,term_pred -> only with same length tokens-wise
    all_comb_len = [(len(t_i),t_i,t_j) for t_i,t_j in itertools.product(sep_term_true,sep_term_predicted) if len(t_i) == len(t_j)]

    # group them by length using dict
    grouped = defaultdict(list) 
    for lterm,true_term,pred_term in all_comb_len: grouped[lterm].append([true_term,pred_term])     # dict[length of term] = [[term_true_1,term_pred_1],[term_true_1,term_pred_2],...]

    correct_pairing_dict  = {}
    for i in sep_term_true: correct_pairing_dict[" ".join(i)] = []
    
    correct_obj_dict  = {}
    for i in sep_term_true: correct_obj_dict[" ".join(i)] = []
    
    # go through term pairing by length 
    for len_term,list_of_term_pair in dict(grouped).items():
        
        tru_term       = [x[0] for x in list_of_term_pair]
        pred_term      = [x[1] for x in list_of_term_pair]

        tru_len        = len(tru_term)
        pred_len       = len(pred_term)

        unique_tru      = np.unique([" ".join(t) for t in tru_term])
        unique_tru_len  = len(unique_tru)
        unique_pred     = np.unique([" ".join(t) for t in pred_term])
        unique_pred_len = len(unique_pred)

        assert unique_tru_len*unique_pred_len == tru_len, f"Noticed repeating terms in Lagrangians"

        # Do checks 
        for iterm_tru, iterm_pred in list_of_term_pair:

            obj_true , complete_true , good_contract_true = sep_objects(" ".join(iterm_tru),checks=True)
            obj_pred , complete_pred , good_contract_pred = sep_objects(" ".join(iterm_pred),checks=True)

            same_objects_bool_ec = same_objects_Q(" ".join(iterm_tru)    ," ".join(iterm_pred))
            same_objects_bool_ys = check_same_objects(" ".join(iterm_tru)," ".join(iterm_pred))
            assert same_objects_bool_ec == same_objects_bool_ys

            # filter by same object
            if same_objects_bool_ys:        
                assert all([len(obj_true) == len(obj_pred) for iobj_true,iobj_pred in zip(obj_true,obj_pred)])
                assert len(obj_true) == len(obj_pred)

                # Correct obj dict
                correct_obj_dict[" ".join(iterm_tru)].append(" ".join(iterm_pred))
                
                if good_contract_true:
                    # do contraction checks here
                    if check_same_contractions(" ".join(iterm_tru)    ," ".join(iterm_pred)):
                        correct_pairing_dict[" ".join(iterm_tru)].append(" ".join(iterm_pred))


        # Each True Term should be paired with exactly 1 Predicted Term, 
        # Do positional based now if there is ambiguity.
        for iterm_tru in unique_tru:
            
            # If more than one term, do position checks
            if len(correct_pairing_dict[iterm_tru]) > 1: 

                true_pos = []
                # for each paired term, redo contractions check but now with positional based checks
                for paired_term in correct_pairing_dict[iterm_tru]:
                    if (check_same_contractions(str(iterm_tru), str(paired_term),True)): true_pos.append(paired_term)
                correct_pairing_dict[iterm_tru] = true_pos
        
        # Check for cases with overlapping elements of the matrix, otherwise take one of them
        for iterm_tru in unique_tru:
            
            # If more than one term, remove cases where its found to exist in other true pairing with only one paired
            if len(correct_pairing_dict[iterm_tru]) > 1: 
                for iterm_tru_others in unique_tru:
                    if (len(correct_pairing_dict[iterm_tru_others]) == 1 
                        and check_same_objects(" ".join(iterm_tru)," ".join(iterm_tru_others))
                        and iterm_tru_others in correct_pairing_dict[iterm_tru]):
                        correct_pairing_dict[iterm_tru].remove(iterm_tru_others)
                
                nmore_cases = len([iterm_trux for iterm_trux in unique_tru  if (len(correct_pairing_dict[iterm_trux]) > 1 
                                                                                and check_same_objects(" ".join(iterm_tru)," ".join(iterm_trux)))])
                if len(correct_pairing_dict[iterm_tru]) > 1 and nmore_cases ==1:
                    correct_pairing_dict[iterm_tru] = [correct_pairing_dict[iterm_tru][-1]]

                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!")
                    print("FOUND CASES WITH AMBIGUITY!")
                    print("correct_pairing_dict[iterm_tru] :",correct_pairing_dict[iterm_tru])
                    print("nmore_cases                     :",nmore_cases                    )
                    for i in correct_pairing_dict[iterm_tru]:
                        print(i)
                    raise KeyError
                # if len(true_pos) > 1:
                #     #print("Repeating Term:",correct_pairing_dict[iterm_tru])
                #     correct_pairing_dict[iterm_tru] = [true_pos[-1]]
                #     assert len(correct_pairing_dict[iterm_tru]) == 1, f"FOUND MORE TERM AFTER POS CHECKS: {len(correct_pairing_dict[iterm_tru])}"


        

    n_term_true        = len(sep_term_true)
    n_term_predicted   = len(sep_term_predicted)

    ncorrect           = len([i for i,j in correct_pairing_dict.items() if len(j) == 1])
    ncorrect_obj       = len([i for i,j in correct_obj_dict.items()     if len(j) >= 1])


    if verbose:
        if ncorrect     != n_term_true:        
            print("Truth Terms with missing in Prediction (all considered):")
            for i,j in correct_pairing_dict.items():
                if len(j) != 1:
                    print(i,j)
            
        if ncorrect_obj != n_term_true:        
            print("Truth Terms with missing in Prediction (by fields):")
            for i,j in correct_obj_dict.items():
                if len(j) == 0:
                    print(i,j)    
        
        if n_term_predicted != n_term_true:
            print("Mismatch on the number of terms:")
            print("n_term_predicted : ", n_term_predicted)
            print("n_term_true      : ", n_term_true)
            if n_term_predicted > n_term_true:
                used_terms = np.concatenate([j for _,j in correct_pairing_dict.items()])
                unpaired_terms = [" ".join(pterm) for pterm in sep_term_predicted if " ".join(pterm) not in used_terms ]
                print("Unpaired Terms:")
                for upterm in unpaired_terms:
                    print(upterm)


    ndiff              = abs(n_term_predicted-n_term_true)
    length_penal       = ndiff         / n_term_true
    obj_score          = ncorrect_obj  / n_term_true
    correct_score      = ncorrect      / n_term_true
    lagrangian_score   = correct_score - length_penal

    assert len([i for i,j in correct_pairing_dict.items() if len(j) > 1]) == 0 , "Should not be more than one match per "  
    assert ncorrect <= ncorrect_obj                                            , "Should not be more than one match per "    

    return  lagrangian_score, obj_score, correct_score, length_penal

def get_lagrangian_score(predicted, expected,verbose=False):
    # 1. Separate terms
    # 2. Check for repeating terms, if repeat mod ID accordingly
    # 3. Get all possible combinations of terms based on number of tokens
    # 4. From those combinations get cases with same objects
    # 5. Check contractions if make sense
    # 6. Check all cases if there are true terms that was paired with more than 1 pred_term
    # 7.    If yes, check pred term is used elsewhere or not with guranteed one term match
    #          If yes, check pred term is used elsewhere or not with guranteed one term match
    #                If yes, delete in many term case
    #                If not, raise Error and modify code is needed
    #          If not,  take one of the prediction
    # 8 Calculate Score 

    assert sanity_check_for_sep_functions(expected) == True
    
    try                     : sanity_check_for_sep_functions(predicted)
    except Exception as e   : print(str(e))
    
    # 1. 
    # Seperate lagrangians to terms
    sep_term_true       , complete_true      = sep_terms(expected,completeness_check=True)
    sep_term_predicted  , complete_predicted = sep_terms(predicted,completeness_check=True)
    sep_term_true_joined                     = [" ".join(i) for i in sep_term_true  ]
    sep_term_predicted_joined                = [" ".join(i) for i in sep_term_predicted  ]

    # check for repeating terms
    assert len(sep_term_true_joined) == len(np.unique(sep_term_true_joined))

    # 2.
    # check for repeating terms in prediction, if so, change to new set of id just in case its a coincidence
    if len(sep_term_predicted_joined) != len(np.unique(sep_term_predicted_joined)):
        uniq_sep = []
        for term,nterm in dict(Counter(sep_term_predicted_joined)).items():
            if nterm != 1:
                for ind in range(nterm): 
                    new_term = replace_ids(term)
                    if len(np.unique([tok for tok in new_term.split() if tok in all_id])) !=0:
                        while (new_term in uniq_sep): new_term = replace_ids(term)
                    uniq_sep.append(new_term)
            else : 
                uniq_sep.append(term)
        sep_term_predicted_joined = uniq_sep
        sep_term_predicted        = [t.split() for t in  uniq_sep]
    
    # if there are terms that dont have ids and they repeat, then its fine. Otherwise check
    if all([check_ids(instr) for instr in sep_term_predicted_joined]):
        assert len(sep_term_predicted_joined) == len(np.unique(sep_term_predicted_joined))

    # 3.
    # Get all combination of term_true,term_pred -> only with same length tokens-wise
    all_comb_len = [(len(t_i),t_i,t_j) for t_i,t_j in itertools.product(sep_term_true,sep_term_predicted) if len(t_i) == len(t_j)]

    # group them by length using dict
    grouped = defaultdict(list) 
    for lterm,true_term,pred_term in all_comb_len: grouped[lterm].append([true_term,pred_term])     # dict[length of term] = [[term_true_1,term_pred_1],[term_true_1,term_pred_2],...]

    correct_pairing_dict  = {}
    for i in sep_term_true: correct_pairing_dict[" ".join(i)] = []
    
    correct_obj_dict  = {}
    for i in sep_term_true: correct_obj_dict[" ".join(i)] = []
    
    # go through term pairing by length 
    for len_term,list_of_term_pair in dict(grouped).items():
        
        tru_term       = [x[0] for x in list_of_term_pair]
        pred_term      = [x[1] for x in list_of_term_pair]

        tru_len        = len(tru_term)
        pred_len       = len(pred_term)

        unique_tru      = np.unique([" ".join(t) for t in tru_term])
        unique_tru_len  = len(unique_tru)
        unique_pred     = np.unique([" ".join(t) for t in pred_term])
        unique_pred_len = len(unique_pred)

        assert unique_tru_len*unique_pred_len == tru_len, f"Noticed repeating terms in Lagrangians"

        # Do checks 
        for iterm_tru, iterm_pred in list_of_term_pair:

            obj_true , complete_true , good_contract_true = sep_objects(" ".join(iterm_tru),checks=True)
            obj_pred , complete_pred , good_contract_pred = sep_objects(" ".join(iterm_pred),checks=True)

            same_objects_bool_ec = same_objects_Q(" ".join(iterm_tru)    ," ".join(iterm_pred))
            same_objects_bool_ys = check_same_objects(" ".join(iterm_tru)," ".join(iterm_pred))
            assert same_objects_bool_ec == same_objects_bool_ys

            # filter by same object
            if same_objects_bool_ys:        
                assert all([len(obj_true) == len(obj_pred) for iobj_true,iobj_pred in zip(obj_true,obj_pred)])
                assert len(obj_true) == len(obj_pred)

                # Correct obj dict
                correct_obj_dict[" ".join(iterm_tru)].append(" ".join(iterm_pred))
                
                if good_contract_true:
                    # do contraction checks here
                    if check_same_contractions(" ".join(iterm_tru)    ," ".join(iterm_pred)):
                        correct_pairing_dict[" ".join(iterm_tru)].append(" ".join(iterm_pred))


        # Each True Term should be paired with exactly 1 Predicted Term, 
        # Do positional based now if there is ambiguity.
        for iterm_tru in unique_tru:
            
            # If more than one term, do position checks
            if len(correct_pairing_dict[iterm_tru]) > 1: 

                true_pos = []
                # for each paired term, redo contractions check but now with positional based checks
                for paired_term in correct_pairing_dict[iterm_tru]:
                    if (check_same_contractions(str(iterm_tru), str(paired_term),True)): true_pos.append(paired_term)
                correct_pairing_dict[iterm_tru] = true_pos
        
        # Check for cases with overlapping elements of the matrix, otherwise take one of them
        for iterm_tru in unique_tru:
            
            # If more than one term, remove cases where its found to exist in other true pairing with only one paired
            if len(correct_pairing_dict[iterm_tru]) > 1: 
                for iterm_tru_others in unique_tru:
                    if (len(correct_pairing_dict[iterm_tru_others]) == 1 
                        and check_same_objects(" ".join(iterm_tru)," ".join(iterm_tru_others))
                        and iterm_tru_others in correct_pairing_dict[iterm_tru]):
                        correct_pairing_dict[iterm_tru].remove(iterm_tru_others)
                
                nmore_cases = len([iterm_trux for iterm_trux in unique_tru  if (len(correct_pairing_dict[iterm_trux]) > 1 
                                                                                and check_same_objects(" ".join(iterm_tru)," ".join(iterm_trux)))])
                if len(correct_pairing_dict[iterm_tru]) > 1 and nmore_cases ==1:
                    correct_pairing_dict[iterm_tru] = [correct_pairing_dict[iterm_tru][-1]]

                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!")
                    print("FOUND CASES WITH AMBIGUITY!")
                    print("correct_pairing_dict[iterm_tru] :",correct_pairing_dict[iterm_tru])
                    print("nmore_cases                     :",nmore_cases                    )
                    for i in correct_pairing_dict[iterm_tru]:
                        print(i)
                    raise KeyError
                # if len(true_pos) > 1:
                #     #print("Repeating Term:",correct_pairing_dict[iterm_tru])
                #     correct_pairing_dict[iterm_tru] = [true_pos[-1]]
                #     assert len(correct_pairing_dict[iterm_tru]) == 1, f"FOUND MORE TERM AFTER POS CHECKS: {len(correct_pairing_dict[iterm_tru])}"


        

    n_term_true        = len(sep_term_true)
    n_term_predicted   = len(sep_term_predicted)

    ncorrect           = len([i for i,j in correct_pairing_dict.items() if len(j) == 1])
    ncorrect_obj       = len([i for i,j in correct_obj_dict.items()     if len(j) >= 1])


    if verbose:
        if ncorrect     != n_term_true:        
            print("Truth Terms with missing in Prediction (all considered):")
            for i,j in correct_pairing_dict.items():
                if len(j) != 1:
                    print(i,j)
            
        if ncorrect_obj != n_term_true:        
            print("Truth Terms with missing in Prediction (by fields):")
            for i,j in correct_obj_dict.items():
                if len(j) == 0:
                    print(i,j)    
        
        if n_term_predicted != n_term_true:
            print("Mismatch on the number of terms:")
            print("n_term_predicted : ", n_term_predicted)
            print("n_term_true      : ", n_term_true)
            if n_term_predicted > n_term_true:
                used_terms = np.concatenate([j for _,j in correct_pairing_dict.items()])
                unpaired_terms = [" ".join(pterm) for pterm in sep_term_predicted if " ".join(pterm) not in used_terms ]
                print("Unpaired Terms:")
                for upterm in unpaired_terms:
                    print(upterm)

    if n_term_predicted > n_term_true:
        nextra              = n_term_predicted-n_term_true
    else:
        nextra = 0

    length_penal       = nextra         / n_term_true
    obj_score          = ncorrect_obj  / n_term_true
    correct_score      = ncorrect      / n_term_true
    lagrangian_score   = correct_score - length_penal

    assert len([i for i,j in correct_pairing_dict.items() if len(j) > 1]) == 0 , "Should not be more than one match per "  
    assert ncorrect <= ncorrect_obj                                            , "Should not be more than one match per "    

    return  lagrangian_score, obj_score, correct_score, length_penal

#################
##### FORMAT #####
#################


def sep_polyobj(inlist):
    sep_sym_ind = []
    for i in np.arange(0,len(inlist),1):
        if inlist[i] in ["SPIN","SU2","U1","SU3","HEL","DAGGER",aei.CONTRACTIONS_TOKEN,"LORENTZ"]+all_id: sep_sym_ind.append(i)
    split_lists = [" ".join(inlist[i:j]) for i, j in zip([0]+sep_sym_ind, sep_sym_ind+[None]) if (inlist[i:j] != [])]
    return split_lists

def check_field_format(inlist,verbose=False):
    polyobj = sep_polyobj(inlist)
    if not (inlist.count("SPIN")   == 1 ): 
        if verbose : print("X"," ".join(inlist))
        return False
    if not (inlist.count("SU2")    <= 1 ): 
        if verbose : print("X"," ".join(inlist))
        return False
    if not (inlist.count("U1")     <= 1 ): 
        if verbose : print("X"," ".join(inlist))
        return False
    if not (inlist.count("SU3")    <= 1 ): 
        if verbose : print("X"," ".join(inlist))
        return False
    if not (inlist.count("HEL")    <= 1 ): 
        if verbose : print("X"," ".join(inlist))
        return False
    if not (inlist.count("DAGGER") <= 1 ): 
        if verbose : print("X"," ".join(inlist))
        return False
    if inlist[-1] not in all_id:
        if verbose : print("X"," ".join(inlist))
        return False

    assert polyobj[0] == aei.FIELD_TOKEN
    for i in polyobj[1:]:
        if "SPIN" in i  : 
            if not ( i in ["SPIN 0","SPIN 1 / 2"])     : 
                if verbose : print("X"," ".join(inlist))
                return False                         
        elif "HEL" in i : 
            if not ( i in ["HEL - 1 / 2","HEL 1 / 2"]) : 
                if verbose : print("X"," ".join(inlist))
                return False                             
        elif "SU3" in i : 
            if not ( i in ["SU3 3","SU3 - 3"])         : 
                if verbose : print("X"," ".join(inlist))
                return False                     
        elif "SU2" in i : 
            if not ( i in ["SU2 3","SU2 2"])           : 
                if verbose : print("X"," ".join(inlist))
                return False                     
        elif "ID" in i  : 
            if not ( i in all_id)                   : 
                if verbose : print("X"," ".join(inlist))
                return False             
        elif "U1" in i  : 
            if not (i.count("/") <= 1):
                if verbose : print("X"," ".join(inlist))
                return False
            if not (i.count("-") <= 1):
                if verbose : print("X"," ".join(inlist))
                return False
            if not (len(i.replace("U1","").replace(" ","")) <= 5):
                if verbose : print("X"," ".join(inlist))
                return False
            try    : eval(i.replace("U1","").replace(" ",""))
            except : 
                if verbose : print("X"," ".join(inlist))
                return False
        elif  i == "DAGGER" : continue
        else: 
            raise KeyError
    
    return True

def check_format(instr,verbose=False):
    terms, completeness = sep_terms(instr,completeness_check=True)

    checks = []
    for iterm in terms:
        id_list    = [i for i in iterm if i in all_id]
        objs       = sep_objects(iterm)
        check_term = []
        iterm_str = " ".join(iterm)
        ncom_a = iterm_str.count(aei.COMMUTATOR_TOKEN_A)
        ncom_b = iterm_str.count(aei.COMMUTATOR_TOKEN_B)
        for iobj in objs:

            # If Single-token-object, it can only be one of the followings
            if len(iobj) == 1:
                if iobj[0] in ["+","-","i",aei.COMMUTATOR_TOKEN_A,aei.COMMUTATOR_TOKEN_B,aei.CONTRACTIONS_TOKEN] : 
                    check_term.append(True)
                else                     : 
                    if verbose: 
                        print("Single-token-object that doesnt look right")
                        print("  Term :",iterm_str)
                        print("  Obj  :"," ".join(iobj))
                    check_term.append(False)
            
            # If the object is CONTRACTIONS, check contractions formats
            elif aei.CONTRACTIONS_TOKEN in iobj:
                if iobj[0] != aei.CONTRACTIONS_TOKEN:
                    cont_check = False
                    if verbose: 
                        print("Contraction doesnt look right: Weird Object Separation")
                        print("  Term :",iterm_str)
                        print("  Obj  :"," ".join(iobj))
                else:    
                    cont_check = check_contractions_format(iobj,id_list,False)
                    if verbose and not cont_check: 
                        print("Contraction doesnt look right")
                        print("  Term :",iterm_str)
                        print("  Obj  :"," ".join(iobj))
                check_term.append(cont_check)

            # Otherwise, it can only be a polytoken-object
            else:
                # If field object
                if aei.FIELD_TOKEN      in iobj: 
                    field_check = check_field_format(iobj,False)
                    if verbose and not field_check: 
                        print("Field Object doesnt look right")
                        print("  Term :",iterm_str)
                        print("  Obj  :"," ".join(iobj))
                    check_term.append(field_check)

                # If Derivative object
                if aei.DERIVATIVE_TOKEN in iobj: 
                    if ((len(iobj) > 5) or len(iobj[1:-1]) > 3
                        or (iobj[0] != aei.DERIVATIVE_TOKEN)
                        or any([i not in in_sym_group_vocab for i in iobj[1:-1]])
                        or (iobj[-1] not in  all_id) ): 
                        if verbose: 
                            print("DERIVATIVE Object doesnt look right")
                            print("  Term :",iterm_str)
                            print("  Obj  :"," ".join(iobj))
                        check_term.append(False)
                    else                : 
                        check_term.append(True)

                # If Sigma object
                if aei.SIGMA_BAR_TOKEN  in iobj:
                    if ((len(iobj)    != 2) 
                        or (iobj[0]  != aei.SIGMA_BAR_TOKEN)
                        or (iobj[-1]  not in  all_id)) : 
                        if verbose: 
                            print("Sigma Object doesnt look right")
                            print("  Term :",iterm_str)
                            print("  Obj  :"," ".join(iobj))
                        check_term.append(False)
                    else                : 
                        check_term.append(True)
        
        if (ncom_a != ncom_b):
            checks.append(False)
            if verbose: 
                print("Unpaired Commutators")
                print("  Term  :", iterm_str)
                print("  ncom_a:",ncom_a)
                print("  ncom_b:",ncom_b)
        elif len(objs[-1]) == 1 and (objs[-1][0] in [aei.COMMUTATOR_TOKEN_A,aei.COMMUTATOR_TOKEN_B]): 
            checks.append(False)
            if verbose: 
                print("Unfilled Commutators")
                print("  Term :", iterm_str)
        else:
            checks.append(all(check_term))
    if (not completeness) and verbose :
        print("Incomplete Terms")
        print("  Term :", iterm_str)
    return checks, completeness

def check_contractions_format(inlist,id_list,verbose=False):
    sep_sym_ind = []
    for i in np.arange(0,len(inlist),1):
        if inlist[i] in ["SU2","SU3","LORENTZ",aei.CONTRACTIONS_TOKEN]: sep_sym_ind.append(i)
    split_lists = [" ".join(inlist[i:j]) for i, j in zip([0]+sep_sym_ind, sep_sym_ind+[None]) if (inlist[i:j] != [])]
    
    # If contain more than CONTRACTIONS_TOKEN token
    if aei.CONTRACTIONS_TOKEN in split_lists[0]  and  split_lists[0] != aei.CONTRACTIONS_TOKEN:  
        if verbose: print("!",inlist)
        return False

    assert  split_lists[0] == aei.CONTRACTIONS_TOKEN , split_lists
    if not(  split_lists.count(aei.CONTRACTIONS_TOKEN) == 1):  
        if verbose: print("!",inlist)
        return False

    for iobj in split_lists[1:]:
        if not( iobj.count("SU2") +  iobj.count("SU3") +  iobj.count("LORENTZ")  == 1  ): 
            if verbose: print("!",inlist)
            return False
        if not( iobj.split()[0] in ["SU2","SU3","LORENTZ"])                             : 
            if verbose: print("!",inlist)
            return False
        if not( all( [i in all_id  for i in iobj.split()[1:]]))                      : 
            if verbose: print("!",inlist)
            return False
        if not( all( [i in id_list  for i in iobj.split()[1:]]))                        : 
            if verbose: print("!",inlist)
            return False
    if inlist[-1] not in all_id :
        if verbose: print("!",inlist)
        return False

    return True

def show_problematic_mass_dimension(instring,check_last_term=True):
    term_string_list = [" ".join(i) for i in sep_terms(instring)]
    phi_string_list = [i.count("FIELD SPIN 0") for i in term_string_list]
    psi_string_list = [i.count("FIELD SPIN 1 / 2") for i in term_string_list]
    der_string_list = [i.count(aei.DERIVATIVE_TOKEN) for i in term_string_list]
    print("phi_string_list: ",phi_string_list)
    print("psi_string_list: ",psi_string_list)
    print("der_string_list: ",der_string_list)
    n=0        
    for iterm, phi_dim,psi_dim,der_dim in zip(term_string_list, phi_string_list,psi_string_list,der_string_list):
        n+=1
        if check_last_term == False and n==len(term_string_list): return True
        mdim = phi_dim+(3/2)*psi_dim+der_dim
        if mdim != 4 and not (   (phi_dim == 3 and psi_dim == 0 and der_dim == 0) # trilinear
                              or (phi_dim == 2 and psi_dim == 0 and der_dim == 0) # mass
                              or (phi_dim == 0 and psi_dim == 2 and der_dim == 0) # mass
                              ) : 
            print(iterm)
            for iobj in sep_objects(iterm):
                print(iobj)
    return None


def check_mass_dimension(instring,verbose=False):
    term_string_list = [" ".join(i) for i in sep_terms(instring)]
    phi_string_list  = [i.count("FIELD SPIN 0") for i in term_string_list]
    psi_string_list  = [i.count("FIELD SPIN 1 / 2") for i in term_string_list]
    der_string_list  = [i.count(aei.DERIVATIVE_TOKEN) for i in term_string_list]
    n=0        
    mdim_list = [] 
    for phi_dim,psi_dim,der_dim,iterm in zip(phi_string_list,psi_string_list,der_string_list,term_string_list):
        mdim = phi_dim+(3/2)*psi_dim+der_dim
        if mdim != 4 and not (   (phi_dim == 3 and psi_dim == 0 and der_dim == 0) # trilinear
                              or (phi_dim == 2 and psi_dim == 0 and der_dim == 0) # mass
                              or (phi_dim == 0 and psi_dim == 2 and der_dim == 0) # mass
                              ) : 
            if verbose: 
                print("term     :",iterm)
                print("phi_dim  :",phi_dim)
                print("psi_dim  :",psi_dim)
                print("der_dim  :",der_dim )
            mdim_list.append(False)
        else:
            mdim_list.append(True)
    return mdim_list


###
###
 
def check_intersection_w_perm(field_string_in,field_string_ref):

    field_string_in   = field_string_in.split(aei.FIELD_TOKEN)[1:]
    field_string_ref  = field_string_ref.split(aei.FIELD_TOKEN)[1:]

    for i in range(len(field_string_in)):
        while field_string_in[i][0] == " ": field_string_in[i] = field_string_in[i][1:]
        while field_string_in[i][-1] == " ": field_string_in[i] = field_string_in[i][:-1]

    for i in range(len(field_string_ref)):
        while field_string_ref[i][0] == " ":    field_string_ref[i] = field_string_ref[i][1:]
        while field_string_ref[i][-1] == " ":    field_string_ref[i] = field_string_ref[i][:-1]

    if len(set(field_string_in).union(set(field_string_ref))) == len(set(field_string_in)) + len(set(field_string_ref)): return True
    else: return False

def check_field_match_w_perm(field_string_in,field_string_ref):

    field_string_in   = field_string_in.split(aei.FIELD_TOKEN)[1:]
    field_string_ref  = field_string_ref.split(aei.FIELD_TOKEN)[1:]

    for i in range(len(field_string_in)):
        while field_string_in[i][0] == " ": field_string_in[i] = field_string_in[i][1:]
        while field_string_in[i][-1] == " ": field_string_in[i] = field_string_in[i][:-1]

    for i in range(len(field_string_ref)):
        while field_string_ref[i][0] == " ":    field_string_ref[i] = field_string_ref[i][1:]
        while field_string_ref[i][-1] == " ":    field_string_ref[i] = field_string_ref[i][:-1]
    
    
    if (len(set(field_string_in).union(set(field_string_ref))) == len(set(field_string_in))) or len(set(field_string_in).union(set(field_string_ref))) == len(set(field_string_ref)): return True
    else                                                                                                                                                                            : return False

def check_existence_w_perm(field_string_in,ref_qn_sets_dat):
    field_string_in = sep_fs_into_qn_sets(field_string_in)
    for irefset in ref_qn_sets_dat:
        if (len(set(field_string_in).union(irefset)) == len(set(field_string_in))) or len(set(field_string_in).union(irefset)) == len(irefset): return True
    return False
 
def get_nterm_w_u1_conservation(instr, check_last_term=True,norm_to_u1terms=False,verbose=False):
    inlist = sep_terms(instr)
    if check_last_term == False: inlist = inlist[:-1]

    nconserved= 0
    nterm = 0
    for term in inlist:
        if norm_to_u1terms and "U1" not in term: 
            continue
        nterm += 1    
        u1_conserved = True
        obj,completeness,good_contraction = sep_objects(term,True)
        u1_obj = [iobj for iobj in obj if ("U1" in iobj and aei.FIELD_TOKEN in iobj)]
        u1_dat = []
        for iobj in u1_obj:
            u1_ind = [n for n,itok in enumerate(iobj) if "U1" == itok]
            try:
                assert len(u1_ind) == 1, (f"{u1_ind} {iobj}")
                u1_ind = u1_ind[0]+1
                iobj = iobj[u1_ind:]
                if "HEL" in iobj:
                    hel_ind = [n for n,itok in enumerate(iobj) if "HEL" == itok]
                    assert len(hel_ind) == 1
                    hel_ind = hel_ind[0]
                    iobj = iobj[:hel_ind]

                if "DAGGER" in iobj:
                    dagger_ind = [n for n,itok in enumerate(iobj) if "DAGGER" == itok]
                    assert len(dagger_ind) == 1
                    dagger_ind = dagger_ind[0]
                    iobj = iobj[:dagger_ind]

                if any([itok in all_id for itok in iobj]):
                    id_ind = [n for n,itok in enumerate(iobj) if itok in all_id]
                    assert len(id_ind) == 1
                    id_ind = id_ind[0]
                    iobj = iobj[:id_ind]
            
            
                if "/"in iobj:
                    split_ind = [n for n,itok in enumerate(iobj) if "/" == itok]
                    if len(split_ind) > 1 : 
                        if verbose: print(" ".join(term),"\n",iobj)
                        u1_conserved = False
                        break
                    #assert len(split_ind) == 1
                    split_ind = split_ind[0]
                    num = " ".join(iobj[:split_ind])
                    denom = " ".join(iobj[split_ind+1:])
                    while " " in num:num = num.replace(" ","")
                    while " " in denom:denom = denom.replace(" ","")
                    if len(num) == 0    :
                        if verbose: print(" ".join(term),"\n",iobj)
                        u1_conserved = False
                        break
                    if len(denom) == 0  :
                        if verbose: print(" ".join(term),"\n",iobj)
                        u1_conserved = False
                        break
                    #print(iobj,num,"/",denom)
                    u1_str = Fraction(eval(num),eval(denom))
                    u1_dat.append(u1_str)

                else:
                    u1_str = " ".join(iobj)
                    while " " in u1_str:u1_str = u1_str.replace(" ","")

                    u1_dat.append(eval(u1_str))
            except:
                if completeness == True and good_contraction == True: 
                    print("term : "," ".join(term))
                    print("iobj : "," ".join(iobj))
                    if verbose: print(" ".join(term),"\n",iobj)
                    u1_conserved = False
                    #raise KeyError
                else : 
                    if verbose: print(" ".join(term),"\n",iobj) 
                    u1_conserved = False

        if sum(u1_dat) != 0:
            if verbose: print(" ".join(term))
            u1_conserved = False
        if u1_conserved == True:
            nconserved+=1
    if norm_to_u1terms :           
        return nconserved, nterm 
    else:
        return nconserved


def get_u1_conservation(instr, verbose=False):
    # Seperate strings to term list
    inlist = sep_terms(instr)
    
    #Initialize
    term_w_u1_conserved = []
    nterm_w_u1          = 0
    last_term_completeneess = None
    last_term               = None
    #Go through terms
    for term in inlist:

        #Ignore terms without U1
        if "U1" not in term: continue
        nterm_w_u1 += 1    

        u1_conserved = True
        #Sep Objects
        obj,completeness,good_contraction = sep_objects(term,True)
        last_term_completeneess = completeness
        last_term = " ".join(term)
        
        #Get Objects with U1 tokens
        u1_obj = [iobj for iobj in obj if ("U1" in iobj and aei.FIELD_TOKEN in iobj)]
        u1_dat = []

        #Loop through Objects with U1 tokens
        for iobj in u1_obj:

            # Get indices of tokens with U1
            u1_ind = [n for n,itok in enumerate(iobj) if "U1" == itok]
            try:
                assert len(u1_ind) == 1, (f"An object should only have one U1 token at most. indices of U1 token in {iobj}: {u1_ind} ")
                u1_ind = u1_ind[0]+1
                
                # Take the part of the object from the U(1) charge onwards
                iobj = iobj[u1_ind:]

                # IF there is a HEL, get it before that
                if "HEL" in iobj:
                    hel_ind = [n for n,itok in enumerate(iobj) if "HEL" == itok]
                    assert len(hel_ind) == 1
                    hel_ind = hel_ind[0]
                    iobj = iobj[:hel_ind]

                # IF there is a DAGGER, get it before that
                if "DAGGER" in iobj:
                    dagger_ind = [n for n,itok in enumerate(iobj) if "DAGGER" == itok]
                    assert len(dagger_ind) == 1
                    dagger_ind = dagger_ind[0]
                    iobj = iobj[:dagger_ind]

                # IF theres is ID in the token, get it before that
                if any([itok in all_id for itok in iobj]):
                    id_ind = [n for n,itok in enumerate(iobj) if itok in all_id]
                    assert len(id_ind) == 1
                    id_ind = id_ind[0]
                    iobj = iobj[:id_ind]
            
            
                # What left should be the U1 charge, and now get the value. 
                # It can either be fraction or integer
                if "/"in iobj:
                    split_ind = [n for n,itok in enumerate(iobj) if "/" == itok]

                    # if there is more than one / in a fraction, it is considered broken U1
                    if len(split_ind) > 1 : 
                        if verbose: print(" ".join(term),"\n",iobj)
                        u1_conserved = False
                        
                        break

                    split_ind = split_ind[0]
                    num = " ".join(iobj[:split_ind])
                    denom = " ".join(iobj[split_ind+1:])
                    while " " in num:num = num.replace(" ","")
                    while " " in denom:denom = denom.replace(" ","")

                    # if its a fraction, it should have one numerator and one denominartor
                    if len(num) == 0    :
                        if verbose: print(" ".join(term),"\n",iobj)
                        u1_conserved = False
                        
                        break
                    if len(denom) == 0  :
                        if verbose: print(" ".join(term),"\n",iobj)
                        u1_conserved = False
                        
                        break

                    u1_str = Fraction(eval(num),eval(denom))
                    u1_dat.append(u1_str)

                else:
                    u1_str = " ".join(iobj)
                    # if int, then there should be an integer in the tokens
                    if not any([i in int_vocab for i in iobj]):
                        if verbose: print(" ".join(term),"\n",iobj)
                        u1_conserved = False
                        
                        break

                    while " " in u1_str:u1_str = u1_str.replace(" ","")

                    u1_dat.append(eval(u1_str))
            except Exception as e:
                print(str(e))
                if completeness == True and good_contraction == True: 
                    print("term : "," ".join(term))
                    print("iobj : "," ".join(iobj))
                    if verbose: print(" ".join(term),"\n",iobj)
                    u1_conserved = False
                    
                    #raise KeyError
                else : 
                    print("incomplete term : "," ".join(term))
                    if verbose: print(" ".join(term),"\n",iobj) 
                    u1_conserved = False
                
                break

        if sum(u1_dat) != 0 and u1_conserved == True:
            if verbose: print(" ".join(term))
            u1_conserved = False

        term_w_u1_conserved.append(u1_conserved) 

    assert len(term_w_u1_conserved) <= nterm_w_u1, f"There cannot be more conserved term than terms with U1 -> nterm_w_u1_conserved{len(term_w_u1_conserved) } , nterm_w_u1:{ nterm_w_u1} , nterm:{ len(inlist)}"
    return term_w_u1_conserved,nterm_w_u1,last_term_completeneess,last_term



#################
##### MISC #####
#################

def get_tags(instr,display_flag=True):
    all_tags = []
    for ispred in (sep_terms(instr)):
        nf = " ".join(ispred).count("FIELD SPIN 1 / 2")
        ns = " ".join(ispred).count("FIELD SPIN 0")
        nder = " ".join(ispred).count(aei.DERIVATIVE_TOKEN)
        tag = ""
        if nf == 2 and ns == 1             : tag="Yukawa"
        if nf == 0 and ns == 3             : tag="Scalar Trilinear"
        if nf == 0 and ns == 4             : tag="Scalar Quartic"
        if nf == 0 and ns == 2 and nder==0 : tag="Scalar Mass"
        if nf == 2 and ns == 0 and nder==0 : tag="Fermion Mass"
        if nf == 0 and ns == 2 and nder!=0 : tag="Scalar Kinetic"
        if nf == 2 and ns == 0 and nder!=0 : tag="Fermion Kinetic"
        if nf == 0 and ns == 0 and nder!=0 : tag="GB Kinetic"
        all_tags.append(tag)

        if display_flag: 
            print("================================================================================================================")
            print(f"{tag} TERM (s.Pred): \n"," ".join(ispred))
            print("================================================================================================================")
        objs_spred = sep_objects(ispred)
        isign_spred = objs_spred[0]
        objs_spred  = objs_spred[1:]
        #if display_flag: print("\t",isign_true)
        ipd = pd.DataFrame({ "objs_spred"      : [" ".join([j for j in k if "ID" not in j] )                                         for k in objs_spred ],
                             "objs_spred(ID)"  : [[j for j in k if "ID" in j]                   if "ID" in " ".join(k) else ""       for k in objs_spred ],
                             })
        ipd = ipd[ipd["objs_spred"].apply(lambda x: aei.CONTRACTIONS_TOKEN not in x)]
        ipd = ipd[["objs_spred"   , "objs_spred(ID)"  ]]
        if display_flag: display(ipd)
        if "[PAD]" in ispred:
            print("[PAD] in TERM:",ispred)
            ispred = [i for i in ispred if i != "[PAD]"]
        if aei.CONTRACTIONS_TOKEN in ispred and "U1" in ispred:
            contract_ind = [ind for ind,i in enumerate(ispred) if i == aei.CONTRACTIONS_TOKEN][0]
            U1_ind = [ind for ind,i in enumerate(ispred) if i == "U1"]
            if any([ind > contract_ind for ind in U1_ind]):
                print("U1 in TERM:",ispred)
                u1_ind_incontrat = [ind for ind,i in enumerate(ispred[contract_ind:]) if i == "U1"][0]+contract_ind
                if [i for i in ispred[u1_ind_incontrat:] if ("ID" not in i) and i != "U1"] != [] : raise KeyError("Found U1 after contraction, and also other contractions after U1 contractions")
                else: ispred = ispred[:u1_ind_incontrat]
        cont_spred   = contraction_parser(" ".join(ispred))

        if cont_spred == []:
            continue
        #if display_flag: print(cont_true)
        icont = pd.DataFrame({ "cont_spred (Group)" : [c[0] for c in cont_spred],
                               "cont_spred (Term)"  : [[s for s in c[1]] for c in cont_spred],
                               "cont_spred (Ind)"   : [["ID"+s.split("ID")[-1] for s in c[1]] for c in cont_spred],
                               })
        if display_flag: 
            print("=============")
            print("CONTRACTIONS:")
            print("=============")
            display(icont)

    return all_tags    
 
def check_u1_conservation(instr,check_last_term=True):

    inlist = sep_terms(instr)
    if check_last_term == False: inlist = inlist[:-1]

    u1_conserved = True
    for term in inlist:
        obj,completeness,good_contraction = sep_objects(term,True)
        u1_obj = [iobj for iobj in obj if ("U1" in iobj and aei.FIELD_TOKEN in iobj)]
        u1_dat = []
        for iobj in u1_obj:
            u1_ind = [n for n,itok in enumerate(iobj) if "U1" == itok]
            assert len(u1_ind) == 1
            u1_ind = u1_ind[0]+1
            iobj = iobj[u1_ind:]
            if "HEL" in iobj:
                hel_ind = [n for n,itok in enumerate(iobj) if "HEL" == itok]
                assert len(hel_ind) == 1
                hel_ind = hel_ind[0]
                iobj = iobj[:hel_ind]

            if "DAGGER" in iobj:
                dagger_ind = [n for n,itok in enumerate(iobj) if "DAGGER" == itok]
                assert len(dagger_ind) == 1
                dagger_ind = dagger_ind[0]
                iobj = iobj[:dagger_ind]

            if any([itok in all_id for itok in iobj]):
                id_ind = [n for n,itok in enumerate(iobj) if itok in all_id]
                assert len(id_ind) == 1
                id_ind = id_ind[0]
                iobj = iobj[:id_ind]
            
            try:
                if "/"in iobj:
                    split_ind = [n for n,itok in enumerate(iobj) if "/" == itok]
                    if len(split_ind) > 1 : 
                        u1_conserved = False
                        break
                    #assert len(split_ind) == 1
                    split_ind = split_ind[0]
                    num = " ".join(iobj[:split_ind])
                    denom = " ".join(iobj[split_ind+1:])
                    while " " in num:num = num.replace(" ","")
                    while " " in denom:denom = denom.replace(" ","")
                    if len(num) == 0    :
                        u1_conserved = False
                        break
                    if len(denom) == 0  :
                        u1_conserved = False
                        break
                    #print(iobj,num,"/",denom)
                    u1_str = Fraction(eval(num),eval(denom))
                    u1_dat.append(u1_str)

                else:
                    u1_str = " ".join(iobj)
                    while " " in u1_str:u1_str = u1_str.replace(" ","")

                    u1_dat.append(eval(u1_str))
            except:
                if completeness == True and good_contraction == True: 
                    print(" ".join(term))
                    print(" ".join(iobj))
                    raise KeyError
                else : u1_conserved = False

        if sum(u1_dat) != 0:
            u1_conserved = False
    return u1_conserved

def get_uniq_fields(instr,verbose=False):
    terms = sep_terms(instr)
    uniq_f = []
    uniq_cf = []
    for iterm in terms:
        for iobj in sep_objects(iterm):
            if aei.FIELD_TOKEN in iobj:
                ifield = [i for i in iobj if i not in all_id]
                if "DAGGER" not in ifield:
                    if " ".join(ifield) not in uniq_f: uniq_f.append(" ".join(ifield))
                else:
                    if " ".join(ifield) not in uniq_cf: uniq_cf.append(" ".join(ifield))
        
    return uniq_f,uniq_cf


def sep_fs_into_qn_sets(infs):
   infs   = infs.split(aei.FIELD_TOKEN)[1:]
   for i in range(len(infs)):
        while infs[i][0] == " ": infs[i] = infs[i][1:]
        while infs[i][-1] == " ": infs[i] = infs[i][:-1]
   return set(infs)


##########################
##### from prev code #####
##########################

def split_terms(lagrangian):
    # Split on ' + ' but keep '- COMMUTATOR' and '- FIELD'
    terms = re.split(r'\s+\+\s+', lagrangian)
    # Remove the '+' sign from the beginning of terms
    terms = [term[1:] if term.startswith('+') else term for term in terms]
    # Further split terms that contain '- COMMUTATOR' or '- FIELD'
    result = []
    for term in terms:
        if '- COMMUTATOR' in term:
            parts = term.split('- COMMUTATOR')
            result.extend([parts[0].strip(), '- COMMUTATOR' + parts[1].strip()])
        elif '- FIELD' in term:
            parts = term.split('- FIELD')
            result.extend([parts[0].strip(), '- FIELD ' + parts[1].strip()])
        else:
            result.append(term.strip())
    
    # Remove any empty terms
    result = [term for term in result if term]
    
    return result

def split_objects(term):
    # Function to split a term into its constituent objects (FIELD, DERIVATIVE, etc.)
    # Steps:
    # 1. Join commutator with derivative for proper splitting
    # 2. Split the term by 'CONTRACTIONS' and keep only the first part
    # 3. Iterate through words, grouping them into objects based on specific keywords
    # 4. Return the list of objects
    
    #join commutator with derivative to split properly
    term = term.replace('COMMUTATOR_A DERIVATIVE', 'COMMUTATOR_A_DERIVATIVE')
    term = term.replace('COMMUTATOR_B DERIVATIVE', 'COMMUTATOR_B_DERIVATIVE')
    # Split the input string by 'CONTRACTIONS' and take only the first part
    relevant_part = term.split('CONTRACTIONS')[0].strip()
    
    # Split the relevant part by 'FIELD' and 'DERIVATIVE'
    objects = []
    current_item = ''
    
    for word in relevant_part.split():
        if word in ['FIELD', 'DERIVATIVE', 'COMMUTATOR_A_DERIVATIVE', 'COMMUTATOR_B_DERIVATIVE', 'SIGMA_BAR']:
            if current_item:
                objects.append(current_item.strip())
            current_item = word
        else:
            current_item += ' ' + word
    
    # Append the last item if it exists
    if current_item:
        objects.append(current_item.strip())
    
    return objects

def same_objects_Q(term1, term2):
    # Function to check if two terms have the same objects
    # Steps:
    # 1. Split both terms into objects
    # 2. Remove IDs from objects
    # 3. Sort the objects
    # 4. Compare the sorted lists of objects
    
    term1_objects = split_objects(term1)
    # remove IDs
    term1_objects = [re.sub('ID\d+', '', obj) for obj in term1_objects]
    # sort objects
    term1_objects.sort()
    
    term2_objects = split_objects(term2)
    # remove IDs
    term2_objects = [re.sub('ID\d+', '', obj) for obj in term2_objects]
    # sort objects
    term2_objects.sort()
    #print("term1_objects    : ",term1_objects)
    #print("term2_objects    : ",term2_objects)
    # check if objects are the same
    if term1_objects == term2_objects:
        return True
    else:
        return False

def contraction_parser(term):
    # Function to parse the contractions part of a term
    # Steps:
    # 1. Extract the contractions part of the term
    # 2. Split contractions into different types (SU3, SU2, LORENTZ)
    # 3. Convert contractions to a list of [type, [elements]] format
    # 4. Replace IDs in contractions with corresponding objects
    
    # get contractions part
    try   :
        relevant_part = term.split('CONTRACTIONS')[1].strip()
    except:
        return []
    # get different contractions
    contractions = []
    current_item = ''

    #print("relevant part: ", relevant_part)

    if len(relevant_part.split()) != 0:
        if (relevant_part.split()[0] not in ['SU3', 'SU2', 'LORENTZ']) :
            relevant_part = 'NO_GROUP ' + relevant_part
    
    #print("relevant part: ", relevant_part)

    for word in relevant_part.split():
        if word in ['SU3', 'SU2', 'LORENTZ', "NO_GROUP"]:
            if current_item:
                contractions.append(current_item.strip())
            current_item = word + "DELIMITER"
        else:
            current_item += ' ' + word

    
    # Append the last item if it exists
    if current_item:
        contractions.append(current_item.strip())
    
    # turn into dictionary with key as contraction type and value as contraction, split by space also
    contractions = [[contraction.split('DELIMITER')[0], contraction.split('DELIMITER')[1].split()] for contraction in contractions]

    
    # get objects
    objects = split_objects(term)

    # replace ID in contractions with corresponging object
    for i, contraction in enumerate(contractions):
       for j, ID in enumerate(contraction[1]):
            for obj in objects:
                if ID in obj:
                    contractions[i][1][j] = obj
    
    return contractions  
    
def same_contractions_Q(term1, term2):
    # Function to check if two terms have the same contractions
    # Steps:
    # 1. Parse contractions for both terms
    # 2. Remove IDs from contractions
    # 3. Sort the contractions
    # 4. Compare the sorted lists of contractions
    
    term1_contractions = contraction_parser(term1)
    # remove IDs from second term in each contraction
    term1_contractions = [[contraction[0], [re.sub('ID\d+', '', obj).strip() for obj in contraction[1]]] for contraction in term1_contractions]

    term2_contractions = contraction_parser(term2)
    # remove IDs
    term2_contractions = [[contraction[0], [re.sub('ID\d+', '', obj).strip() for obj in contraction[1]]] for contraction in term2_contractions]

    # sort everything in both terms
    term1_contractions.sort()
    term2_contractions.sort()

    # check if contractions are the same
    if term1_contractions == term2_contractions:
        return True
    else:
        return False

# def get_lagrangian_score(predicted, expected):
    # Function to calculate a score based on the similarity of predicted and expected Lagrangian terms
    # Steps:
    # 1. Split predicted and expected Lagrangians into terms
    # 2. Calculate length punishment for difference in number of terms
    # 3. Compare objects and contractions of each predicted term with expected terms
    # 4. Calculate scores for object matches and contraction matches
    # 5. Determine completely correct terms (correct objects AND correct contractions)
    # 6. Calculate final score and normalized mistake rates
    
    # Split the predicted and expected Lagrangians into individual terms
    predicted_terms = split_terms(predicted)
    expected_terms = split_terms(expected)
  
    # Penalize for difference in number of terms
    length_punishment = abs(len(predicted_terms) - len(expected_terms))

    object_scores = []
    contraction_scores = []
    
    for predicted_term in predicted_terms:
        all_object_scores = []
        for expected_term in expected_terms:
            # Check if objects in predicted term match any expected term
            all_object_scores.append(same_objects_Q(predicted_term, expected_term))
        
        # Find positions where objects match
        correct_objects_positions = [i for i, answer in enumerate(all_object_scores) if answer]

        if len(correct_objects_positions) > 0:
            object_scores.append(True)
            # If objects match, check contractions
            all_contraction_scores = [same_contractions_Q(predicted_term, expected_terms[i]) 
                                      for i in correct_objects_positions]
            correct_contractions = any(all_contraction_scores)
            contraction_scores.append(correct_contractions)
        else:
            object_scores.append(False)
            contraction_scores.append(False)

    # Determine completely correct terms (correct objects AND correct contractions)
    correct_terms = [a and b for a, b in zip(object_scores, contraction_scores)]

    # Calculate final score
    lagrangian_score = (sum(correct_terms) - length_punishment) / len(expected_terms)

    if lagrangian_score != 1:
        normalized_object_mistakes = (len(expected_terms)-sum(object_scores))/len(expected_terms)
        normalized_contraction_mistakes = (len(expected_terms)-sum(contraction_scores))/len(expected_terms)
    else:
        normalized_object_mistakes = 0
        normalized_contraction_mistakes = 0
        
    return lagrangian_score, normalized_contraction_mistakes, normalized_object_mistakes, sum(correct_terms) / len(expected_terms), length_punishment / len(expected_terms)
