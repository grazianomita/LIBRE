import warnings
warnings.filterwarnings("ignore")

import itertools
import csv
import timeit
import random
import sys
import os
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from boolean_rule_set.libre import LIBRE
from sklearn.model_selection import train_test_split

'''
Utility functions
'''

def read_file(filename, delimiter=';'):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=delimiter)) #reads csv into a list of lists
    return data

def compute_num_rules_and_atoms(rule_set):
    '''
    It returns number of rules, and the number of atoms per rule.
    '''
    num_rules = len(rule_set)
    atom_list = [] 	# list of num_atoms, corresponding to each rule	

    for r in range(num_rules):
        rule = rule_set[r]
        num_atoms = 0
        for c in rule:
            num_atoms += len(c)
        atom_list.append(num_atoms)

    return num_rules, atom_list


'''
Data parameters
'''
data_path = './data/heart.csv'
with_missing_values = False
missing_value_symbol = '?'
feature_names = ['age', 'sex', 'chestPainType', 'restingBloodPressure', 'serumCholestoral', 'fastingBloodSugar', 'restingElectrocardiographicResults', 'maximumHeartRateAchieved', 'exerciseInducedAngina', 'STDepressionInducedByExerciseRelativeToRest', 'slopeOfThePeakExercise', 'numberOfMajorVessels', 'thal']
categorical= [False, True, True, False, False, True, False, False, True, False, False, False, True]
min_values = [None, None, None, None, None, None, None, None, None, None, None, None, None] # None, if we do not know


'''
Model parameters
'''
parallel = True
seed = 17

# Discretization parameters
discretization_threshold = 4.6

# Rule generation parameters
search_heuristic = 'H1'
n_estimators = 10
n_features = 3
with_replacement = False
with_filtering = True

# Rule simplification
min_support = 0
top_K = 50
simplify_method = 'WSC'
alpha = .7
max_pred_rules = 10

def run():
	data = read_file(data_path)
	x = [record[:-1] for record in data]
	y = [int(float(record[-1])) for record in data]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
	
	model = LIBRE()
	# 1) Input records are first converted to a suitable binary format.
	model.fit(
		x_train, 
		y_train, 
		categorical, 
		feature_names, 
		min_values, 
		with_missing_values=with_missing_values, 
		missing_value_symbol="?", 
		chi2_threshold=discretization_threshold, 
		max_intervals=0
	)
	# 2) Run n_estimators weak learners on subset of n_features features to get a boundary set (and a rule-set).
	model.run(
		n_estimators=n_estimators, 
		n_features=n_features, 
		with_replacement=with_replacement, 
		heuristic=search_heuristic, 
		parallel=parallel
	)
	# 3) Simplify the boundary previously learned.
	model.simplify(
		with_filtering=with_filtering, 
		min_lsup=min_support, 
		top_K=top_K, 
		method=simplify_method, 
		alpha=alpha
	)
	
	rule_set = model.get_rule_set()
	num_rules, atom_list = compute_num_rules_and_atoms(rule_set)
	for pred_rules in range(1, max_pred_rules+1):
		predictions, _ = model.predict(x_test, top=pred_rules)
		accuracy = accuracy_score(y_test, predictions)
		f1score = f1_score(y_test, predictions)
		precision = precision_score(y_test, predictions)
		recall = recall_score(y_test, predictions, pos_label=1)
		tnr = recall_score(y_test, predictions, pos_label=0)
		num_rules = pred_rules if num_rules >= pred_rules else num_rules
		num_atoms = np.mean(atom_list[:pred_rules])
		print('> test_accuracy: {:02.2f}'.format(accuracy))
		print('> test_f1score: {:02.2f}'.format(f1score))
		print('> test_precision: {:02.2f}'.format(precision))
		print('> test_recall: {:02.2f}'.format(recall))
		print('> test_tnr: {:02.2f}'.format(tnr))
		print('> test_num_rules: {}'.format(pred_rules))
		print('> test_num_atoms: {}'.format(num_atoms))
		print()


random.seed(seed)
np.random.seed(seed)
run()




