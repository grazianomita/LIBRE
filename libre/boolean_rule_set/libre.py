import sys
import timeit
import numpy as np
import random
import json
import itertools
from chi_merge_encoder.cme import ChiMergeEncoder
from boolean_rule_set.brs import BooleanRuleSet

class LIBRE:
	'''
	LIBRE is an ensemble method to learn interpretable rules from data. Weak learners use a bottom-up approach based on monotone Boolean function synthesis -- BRS. 
	Input records are first discretized and converted to binary strings by using ChiMergeEncoder. BRS is then applied on subsets of input features. The output is combined with a simple union to get a boundary set, and finally simplified (we use weighted set cover). From the simplified boundary, a rule set is extracted and used to classify new records.
	For more details refer to section 3.2 and 3.3 in Mita, 2020 - "LIBRE: Learning Interpretable Boolean Rule Ensembles".
	'''
	
	def __init__(self):
		'''
		It initialize the boundary and the ruleset.
		'''
		self.__boundary = []# Boundary
		self.__boundary_statistics = [] # Boundary statistics, one for each a in A
		self.__simplified_boundary = [] # Simplified boundary
		self.__rule_set = [] # Lists the found rules in terms of features and ranges
		self.feature_names = []
		self.categorical = []
		self.missing_value_symbol = '?'
		self.with_missing_values = False
		self.__feature_domain_sizes = []
		self.__discretized_ranges = []
		self.__encoder = ChiMergeEncoder()

	def fit(self, features, labels, categorical, feature_names, min_values, with_missing_values=True, missing_value_symbol="?", chi2_threshold=4.6, max_intervals=0):
		'''
		It discretizes input features and converts them into binary format.
		
		Parameters:
		------------
			@ "features": List of features
			@ "labels": List of labels: 0 and 1 only are accepted
			@ "categorical": List of True/False values. True if the corresponding feature is categorical, False otherwise
			@ "feature_names": List of names for the features in the same order are received
			@ "min_values": List of min_values. None if the corresponding features is categorical or the min value is not known
			@ "missing_value_symbol": Symbol used in the dataset to encode missing values
		'''
		self.feature_names = feature_names
		self.categorical = categorical
		self.missing_value_symbol = missing_value_symbol
		self.with_missing_values = with_missing_values
		self.__feature_domain_sizes = []
		self.__discretized_ranges = [] # For each feature, it lists the discretized ranges. Ex: [[(0,13.1), (13.1,40)], [(0,.5), (.5,1)]]
		
		labels = [int(float(l)) for l in labels] # Labels are converted to integers.
		if len(set(labels)) != 2:
			print('Error: More than 2 labels have been found!')
			sys.exit(1)
		
		self.__encoder = ChiMergeEncoder()
		self.__encoder.fit(features, labels, categorical, min_values, chi2_threshold=chi2_threshold, with_missing_values=with_missing_values, missing_value_symbol=missing_value_symbol)
		binary_features = self.__encoder.transform(features, o_format='IOHE_STR')
		self.T = self.__extract_T(binary_features, labels)
		self.F = self.__extract_F(binary_features, labels)
		self.number_of_positive_records = len(self.T)
		self.number_of_negative_records = len(self.F)
		self.__discretized_ranges = self.__encoder.get_discretized_ranges()
		self.__feature_domain_sizes = self.__encoder.get_discrete_sizes()

	def run(self, n_estimators=10, n_features=2, with_replacement=True, heuristic="H1", parallel=True, selected_features=[]):
		'''
		It runs BRS k times considering different combinations of input features. Since BRS runs on filtered/reduced datasets, the corresponding boundaries must be modified to take into account the filtered features. Boundaries are then combined and stored in self.__boundary.

		Parameters:
		------------
			@ "n_estimators": Maximum number of estimators we run
			@ "n_features": Number of features each estimator runs on
			@ "with_replacement": if True, the random selection of features is done with replacement
			@ "heuristic": H1, or H2. H1 by default
			@ "parallel": If True, the code inside the single estimators run in parallel. Estimators still run sequentially
			@ "selected_features": List of K feature lists. Each ensemble runs on the kth feature list
		'''
		if (len(self.T) == 0 and len(self.F) == 0):
			print('Please, execute fit() before run()!')
			sys.exit(1)
		
		if len(selected_features) == 0:
			self.selected_features = list()
	
			# 1) Randomly select n_features.
			for k in range(n_estimators):
				if with_replacement:
					self.selected_features.append(sorted(list(set([random.randint(0, len(self.feature_names)-1) for i in range(n_features)])), key=int))
				else:
					self.selected_features.append(sorted(random.sample(range(len(self.feature_names)), n_features)))
		else:
			self.selected_features = selected_features

		# 2) For each random set of features.
		self.__boundary = [] # It contains all the results of sc [A1, ..., AN]
		self.__boundary_statistics = {}
		expanded_boundary_k_statistics = []
		self.__BRS = BooleanRuleSet(heuristic=heuristic, parallel=parallel) # Istantiate BRS
		for k in range(n_estimators):
			sf = self.selected_features[k]
			skip_flag = False
			# Check if the current list has been already processed. # If so, the current list is skipped.
			for i in range(k):
				if self.selected_features[i] == sf:
					skip_flag = True
					break
			if skip_flag:
				continue
			filtered_T = self.__filter_features(self.T, sf)
			filtered_F = self.__filter_features(self.F, sf)
			
			# 2a) At this point we can finally run BRS.
			self.__BRS.fit(filtered_T, filtered_F, np.array(self.__feature_domain_sizes)[sf])
			expanded_boundary_k_statistics.append(self.__expand_boundary(self.__BRS.get_boundary_statistics(), sf))

		self.__boundary_statistics = {boundary_point:ebks_k[boundary_point] for ebks_k in expanded_boundary_k_statistics for boundary_point in ebks_k}
		self.__boundary = list(self.__boundary_statistics)
		self.__simplified_boundary = list(self.__boundary)

		if len(self.__boundary) == 0:
			print('Resulting boundary is empty!')
		else:
			self.extract_rule_set()
	
	'''
	def simplify(self, with_filtering=True, min_lsup=0, top_K=200, method='WSC', alpha=.7):
		\'''
		It performs a simplification over the combination of the expanded boundaries generated by the single estimators. We need first to set T and F to the full (in terms of input features) dataset. Then, we can set the boundary and we can finally run BRS simplify.

		Parameters:
		------------
			@ "with_filtering": True, if filtering is applied before simplification
			@ "min_lsup": minimum local support. Used only if with_filtering is True
			@ "top_K": top rules selected for simplification. Used only if with_filtering is True
			@ "method": weighted set cover (WSC) only, for the moment
			@ "alpha": weight associated to true positives
		\'''
		self.__BRS.set_T(self.T)
		self.__BRS.set_F(self.F)
		self.__BRS.set_boundary(self.__boundary)
		self.__BRS.set_boundary_statistics(self.__boundary_statistics)
		self.__BRS.simplify(with_filtering=with_filtering, min_lsup=min_lsup, top_K=top_K, method=method, alpha=alpha)
		self.__simplified_boundary = self.__BRS.get_boundary()
		self.extract_rule_set()
	'''
	
	def simplify(self, with_filtering=True, min_lsup=0, top_K=200, method='WSC', alpha=.7):
		'''
		It performs a simplification over the combination of the expanded boundaries generated
		by the single estimators. We need first to set T and F to the full (in terms of input
		features) dataset. Then, we can set the boundary and we can finally run BRS simplify.

		Parameters:
		------------
			@ "with_filtering": True, if filtering is applied before simplification.
			@ "min_lsup": minimum local support. Used only if with_filtering is True.
			@ "top_K": top rules selected for simplification. Used only if with_filtering is True.
			@ "method": weighted set cover (WSC) only, for the moment.
			@ "alpha": weight associated to true positives.
		'''
		if alpha>1 or alpha <0:
			print('Error: alpha must be in [0,1]')
			sys.exit(1)
		self.__simplified_boundary = [] # Simplified boundary.
		self.__min_lsup = min_lsup
		# 1) Compute parameters for weighted set cover
		parameters = self.__compute_parameters_for_weighted_set_cover()
		# 2) Parameters are sorted according to (exclusiveness, local_support, number_of_zeros). Only the top_K rules are taken.
		# print('> Sorting results...')
		start = timeit.default_timer()
		if with_filtering:
			import operator
			parameters = sorted(parameters, key=operator.itemgetter(5, 3, 6), reverse=True)[:top_K]
		# print('> done in', timeit.default_timer()-start, 'seconds.')
		# If there are no rules satisfying the minimum support, return.
		if len(parameters) == 0:
			return
		# 3) We are finally ready to run set covering. We need to prepare a data structure first.
		dict_A = {}
		A_tmp = []
		for x in parameters:
			a = x[0]
			A_tmp.append(a)
			dict_A[a] = {}
			dict_A[a]['T'] = x[1]
			dict_A[a]['F'] = x[2]
			dict_A[a]["number_of_zeros"] = x[6]
		# print('> Running WSC...')
		start = timeit.default_timer()
		self.__simplified_boundary = self.__weighted_set_cover(A_tmp, dict_A, alpha=alpha)
		self.extract_rule_set()	
	
	def predict(self, X, top=10):
		'''
		It receives a list of records and returns the corresponding labels according to the boundary extracted from the data. A positive label for sample x is predicted if there is at least one  boundary point a in A that covers x. For each sample, it returns the label and the set of rules that fired, if any.

		Parameters:
		------------
			@ X: List of records to process
		'''
		# Predictions are always done according to the simplified boundary.
		# If the boundary set is empty all records are predicted as negative.
		if (len(self.__simplified_boundary) == 0):
			return [0]*len(X), [[]]*len(X)

		if top >= len(self.__simplified_boundary):
			top = len(self.__simplified_boundary)

		# Convert input records in the right format.
		X_ = self.__encoder.transform(X, o_format='IOHE_STR')

		# Extract the rule set if it has not been extracted yet.
		if len(self.__rule_set) == 0:
			self.extract_rule_set()

		predictions = []
		rules = []
		for x_ in X_:
			at_least_one_rule = False
			x_rules = []
			for i in range(0, top):
				a = self.__simplified_boundary[i]
				if self.__covers(a, x_):
					at_least_one_rule = True
					x_rules.append(self.__rule_set[i])
			if at_least_one_rule == True:
				predictions.append(1)
			else:
				predictions.append(0)
			rules.append(x_rules)

		return predictions, rules

	'''
	def combine_boundary_points(self, boundary):
		# Sometimes, it is possible that some boundary points can be combined together.
		# For example: 
		# if F1 in [None, 0.5) -> 1
		# if F1 in [0.5, 0.6) -> 1
		# may be combined in one unique rule: if F1 in [None, 0.6) -> 1

		# Each boundary point is assigned to a bucket according to which features it uses.
		boundary_buckets = {}
		for a in boundary:
			current_key = ''
			start_idx = 0
			end_idx = 0
			for i in range(0, len(self.__feature_domain_sizes)):
				end_idx += self.__feature_domain_sizes[i]
				a_i = a[start_idx:end_idx] # Binary substring related to the current feature.
				start_idx = end_idx
				if a_i.count('1') == 0:
					continue
				else:
					current_key += str(i)
			if current_key not in boundary_buckets:
				boundary_buckets[current_key] = [a]
			else:
				boundary_buckets[current_key].append(a)

		# Boundary points in the same bucket are combined
		boundary_2 = []
		for key in boundary_buckets:
			# print(key, boundary_buckets[key])
			if len(boundary_buckets[key]) == 1:
				boundary_2 += boundary_buckets[key]
			else:
				a_2 = ''
				for i in range(sum(self.__feature_domain_sizes)):
					done = False
					for j in range(len(boundary_buckets[key])):
						# print(i, j, rule_buckets[key][j][i])
						if boundary_buckets[key][j][i] == '0':
							a_2 += '0'
							done = True
							break
					if done == False:
						a_2 += '1'
				boundary_2.append(a_2)

		return [self.extract_rule(a) for a in boundary_2]
	'''

	'''
	def flatten_rule_set(self, rule_set):
		# Finally, we can flatten the rules.

		f_rule_set = list()
		for r in rule_set:
			r_ = []
			for c_r in r:
				if len(c_r) == 0:
					r_.append([[]])
				else:
					r_.append(c_r)
			for x in (list(itertools.product(*r_))):
				f_rule_set.append(x)
		avg_atoms = 0
		for f_r in f_rule_set:
			count = 0
			for c in f_r:
				if len(c) != 0:
					count += 1
			avg_atoms += count

		print('flattened_rule_set:')
		for x in f_rule_set:
		  print(x)
		print(len(f_rule_set), float(avg_atoms)/len(f_rule_set))

		if len(f_rule_set) > 0:
			return len(f_rule_set), float(avg_atoms)/len(f_rule_set)
		else:
			return 0, None
		return f_rule_set
	'''

	def __compute_parameters_for_weighted_set_cover(self):
		'''
		For every element of the boundary set, it computes a set of parameters needed by weighted set cover.
		'''
		parameters = []
		for boundary_point in self.__boundary_statistics:
			indexes_of_covered_elements_in_T = set(self.__boundary_statistics[boundary_point][0])
			indexes_of_covered_elements_in_F = set(self.__boundary_statistics[boundary_point][1])
			number_of_covered_elements_in_T = len(indexes_of_covered_elements_in_T)
			number_of_covered_elements_in_F = len(indexes_of_covered_elements_in_F)
			local_support_positive_class = number_of_covered_elements_in_T / self.number_of_positive_records
			local_support_negative_class = number_of_covered_elements_in_F / self.number_of_negative_records
			exclusiveness = local_support_positive_class / (local_support_positive_class + local_support_negative_class)
			number_of_zeros = self.__boundary_statistics[boundary_point][2]
			# Filtering
			if local_support_positive_class > local_support_negative_class and local_support_positive_class > self.__min_lsup:
				parameters.append([
					boundary_point, indexes_of_covered_elements_in_T, indexes_of_covered_elements_in_F, local_support_positive_class, local_support_negative_class, exclusiveness, number_of_zeros]
				)
		return parameters
	
	def __weighted_set_cover(self, A_tmp, dict_A, alpha=.7):
		'''
		It runs a greedy implementation of weighted set cover.

		Parameters:
		------------
			@ "A_tmp": boundary.
			@ "dict_A": dictionary containing statistics on boundary points.
			@ "alpha": weight associated to true positives.
		'''
		A_star = []
		S = set(self.T)
		while len(S) > 0:
			best_a = self.__get_best_a(A_tmp, dict_A, alpha)
			# Remove from S all positive elements covered by best_A.
			s_to_be_removed = [s for s in dict_A[best_a]['T']]
			for s in s_to_be_removed:
				S.discard(self.T[s])
			A_tmp.remove(best_a)
			A_star.append(best_a)
			# Update T and F for the remaining boundary points.
			a_to_be_deleted = []
			for a in A_tmp:
				for t in list(dict_A[best_a]['T']):
					if t in dict_A[a]['T']:
						dict_A[a]['T'].discard(t)
				for f in list(dict_A[best_a]['F']):
					if f in dict_A[a]['F']:
						dict_A[a]['F'].discard(f)
				if len(dict_A[a]['T']) == 0:
					a_to_be_deleted.append(a)
			dict_A.pop(best_a, None)
			# Remove redundant elements in the boundary.
			for a in a_to_be_deleted:
				A_tmp.remove(a)
				del(dict_A[a])
			if len(A_tmp) == 0:
				break
		return list(A_star)
	
	def __get_best_a(self, A, dict_A, alpha):
		'''
		Select the element a in A that maximizes in lexicographic order the tuple:
		(alpha*|T| - (1-alpha)*|F|, number_of_zeros)
	
		Parameters:
		------------
			@ "dict_A": dictionary containing, for each a in A, the parameters used to choice the best a.
		'''
		best_a = A[0]
		for a in A:
			z_a = len(dict_A[a]['T'])*alpha - len(dict_A[a]['F'])*(1-alpha)
			z_best_a = len(dict_A[best_a]['T'])*alpha - len(dict_A[best_a]['F'])*(1-alpha)
			if z_a > z_best_a:
				best_a = a
			elif z_a == z_best_a:
				if dict_A[a]["number_of_zeros"] > dict_A[best_a]["number_of_zeros"]:
					best_a = a
		return best_a

	def __expand_boundary(self, boundary_statistics, selected_features):
		'''
		After the execution of BRS on a subset of features, the resulting boundary elements will not have the size of the original dataset. Therefore, boundary elements need to be expanded by adding zeros, corresponding to the missing features.
		
		Parameters:
		------------
			@ "boundary": boundary to be expanded
			@ "selected_features": list of features used to build "boundary", indexes from 0 to #features-1.
		'''
		expanded_boundary_statistics = {}

		for element in boundary_statistics:
			cov_T = boundary_statistics[element][0]
			cov_F = boundary_statistics[element][1]
			number_of_zeros = boundary_statistics[element][2]
			start_idx = 0
			end_idx = 0
			expanded_element = ''
			i = 0
			flag = True
			for f in range(len(self.feature_names)):
				if f in selected_features:
					start_idx = end_idx
					end_idx += self.__feature_domain_sizes[f]
					expanded_element += element[start_idx:end_idx]
				else:
					expanded_element += '0'*self.__feature_domain_sizes[f]
					number_of_zeros += self.__feature_domain_sizes[f]
			expanded_boundary_statistics[expanded_element] = [cov_T, cov_F, number_of_zeros]

		return expanded_boundary_statistics

	def __filter_features(self, data, selected_features):
		'''
		It filters the original data by selecting only the input features received as parameter.
		
		Parameters:
		------------
			@ "data": input binary dataset
			@ "selected_features": list of the indexes of the features to be kept
		'''
		filtered_data = []      # dataset containing only the filtered features

		for record in data:
			start_idx = 0
			end_idx = 0
			filtered_record = ''
			i = 0
			for f in range(len(self.feature_names)):
				start_idx = end_idx
				end_idx += self.__feature_domain_sizes[f]
				if f == selected_features[i]:
					filtered_record += record[start_idx:end_idx]
					i += 1
					if i >= len(selected_features):
						break
			filtered_data.append(filtered_record)

		return filtered_data

	def __extract_T(self, binary_features, labels):
		'''
		It stores in self.T all the binary features with positive (1) label.
		
		Parameters:
		------------
			@ "binary_features": list of binary records
			@ "labels": list of labels associated to binary_features
		'''
		return [binary_features[i] for i in range(0, len(labels)) if labels[i] == 1]

	def __extract_F(self, binary_features, labels):
		'''
		It stores in self.T all the binary features with negative (0) label.
		
		Parameters:
		------------
			@ "binary_features": list of binary records
			@ "labels": list of labels associated to binary_features
		'''
		return [binary_features[i] for i in range(0, len(labels)) if labels[i] == 0]

	def __covers(self, x, y):
		'''
		It checks if x covers y (x <= y).
		x covers y if x OR y = y.

		Parameters:
		------------
			@ "x": binary string
			@ "y": binary string
		'''
		for i in range(0, len(x)):
			xi_or_yi = '0'
			if x[i] == '1' or y[i] == '1':
				xi_or_yi = '1'
			if xi_or_yi != y[i]:
				return False
		return True

	def extract_rule_set(self):
		'''
		rule_set is a list of size boundary size. Each element of the rule_set is itself a list of conditions applied to each feature.
		Example: rule_set = [rule_0, rule_1] where rule_0 = [[Feature1], ..., [FeatureN]].
		'''
		self.__rule_set = rule_set = []
		for a in self.__simplified_boundary:
			rule = self.extract_rule(a)
			rule_set.append(rule)

	def get_rule_set_from_boundary(self, boundary):
		'''
		rule_set is a list of size boundary size. Each element of the rule_set is itself a list of conditions applied to each feature.
		Example: rule_set = [rule_0, rule_1] where rule_0 = [[Feature1], ..., [FeatureN]].
		
		Parameters:
		------------
			@ "boundary": boundary set used to extract the ruleset
		'''
		rule_set = []
		for a in boundary:
			rule = self.extract_rule(a)
			rule_set.append(rule)
		return rule_set

	def extract_rule(self, a):
		'''
		It receives a binary element of the boundary set and returns a list in the following format: [F1, ..., FN] where Fi = [(min, max), ..., (min, max)] are the values (expressed in terms of intervals) that feature Fi can take. Contiguous intervals are automatically combined for non categorical features.

		Parameters:
		------------
			@ "a": boundary point we extract the rule from
		'''
		rule = []
		start_idx = 0
		end_idx = 0
		for i in range(0, len(self.__feature_domain_sizes)):
			end_idx += self.__feature_domain_sizes[i]
			a_i = a[start_idx:end_idx] # Binary substring related to the current feature.
			start_idx = end_idx
			# If a_i is all zeros means that features_i is not considered.
			conditions_on_current_feature = []
			if a_i.count('1') == 0:
				rule.append(conditions_on_current_feature)
				continue
			# We analyze the binary substring related to the current feature to see which intervals are involved.
			# Consecutive intervals (for non categorical features only) are combined in single intervals.
			for j in range(0, len(a_i)):
				if a_i[j] == '0': # and self.__discretized_ranges[i][j][0] != None:
					if len(conditions_on_current_feature) > 0:
						if conditions_on_current_feature[-1][1] == self.__discretized_ranges[i][j][0]:
							conditions_on_current_feature[-1][1] = self.__discretized_ranges[i][j][1]
						else:
							conditions_on_current_feature.append([self.__discretized_ranges[i][j][0], self.__discretized_ranges[i][j][1]])
					else:
						conditions_on_current_feature.append([self.__discretized_ranges[i][j][0], self.__discretized_ranges[i][j][1]])
			rule.append(conditions_on_current_feature)
		
		return rule

	def print_rule_set(self):
		'''
		It prints the rule_set.
		'''
		if len(self.__rule_set) == 0:
			self.extract_rule_set()

		for i in range(0, len(self.__rule_set)):
			rule = self.__rule_set[i]
			print("R" + str(i))
			self.print_rule(rule)

	def print_rule(self, r):
		'''
		It prints a given rule.

		Parameters:
		------------
			@ "r": rule to print
		'''
		for j in range(0, len(r)):
			feature_element = r[j]
			if len(feature_element) > 0:
				if len(self.feature_names) == 0:
					print("\tF" + str(j) + ' IN:')
				else:
					print("\t" + self.feature_names[j] + ' IN:')
			for range_element in feature_element:
				print("\t\t(" + str(range_element[0]) + ', ' + str(range_element[1]) + ')')

	def get_boundary(self):
		'''
		It returns the boundary set.
		'''
		return self.__simplified_boundary

	def get_rule_set(self):
		'''
		It returns the rule set.
		'''
		if len(self.__rule_set) == 0:
			self.extract_rule_set()

		return self.__rule_set

	def export_parameters(self, top):
		'''
		It builds a dictionary storing the main parameters of LIBRE.
		'''
		parameters = self.__encoder.export_parameters()
		parameters['feature_names'] = self.feature_names
		parameters['A'] = self.__boundary
		parameters['A_star'] = self.__simplified_boundary
		parameters['top'] = top
		
		return parameters

	def import_parameters(self, parameters):
		'''
		It receives a dictionary containing the parameters of the model and import them.
		
		Parameters:
		------------
			@ "parameters": {
				'with_missing_values':False,
				'missing_value_symbol':'?',
				'number_of_distinct_labels':2,
				'number_of_features':3,
				'categorical':[True, True, False],
				'feature_domain_sizes':[2, 2, 3],
				'discretized_ranges':[[('gra', 'gra'), ('fra', 'fra')], [('M', 'M'), ('F', 'F')], [(0, 150), (150, 170), (170, 210)]]
				'categorical_mappings':['0': {'gra':0, 'fra':1}, '1':{'M':0, 'F':1}]
				'feature_names':['name', 'sex', 'height']
				'A': ['0000100']
			}
		'''
		keywords = ['with_missing_values', 'missing_value_symbol', 'number_of_distinct_labels', 'number_of_features', 'categorical', 'feature_domain_sizes', 'discretized_ranges', 'categorical_mappings', 'feature_names', 'A']

		for key in keywords:
			if key not in parameters:
				print('> Error: Parameter', key, 'is missing')
				sys.exit(3)

		self.__encoder.import_parameters(parameters)
		self.categorical = parameters['categorical']
		self.__feature_domain_sizes = parameters['feature_domain_sizes']
		self.__discretized_ranges = parameters['discretized_ranges']
		self.feature_names = parameters['feature_names']
		self.__boundary = parameters['A']
		self.__simplified_boundary = parameters['A_star']

		return parameters['top']

	def import_parameters_from_json(self, filename):
		'''
		It imports LIBRE parameters stored in a json file.

		Parameters:
		------------
			@ "filename": path of the file containing the parameters
		'''
		with open(filename, 'r') as infile:
			parameters = json.load(infile)
		self.import_parameters(parameters)

		return parameters['top']

	def export_parameters_to_json(self, filename, top):
		'''
		It exports model parameters to a json file.

		Parameters:
		------------
			@ "filename": path of the output file
		'''
		parameters = self.export_parameters(top)
		with open(filename, 'w+') as outfile:
			json.dump(parameters, outfile)

	@staticmethod
	def test():
		X = [['graziano', '26', 'M', '176'],
			 ['giulio', '18', 'M', '191'],
			 ['?', '29', 'F', '156'],
			 ['francesca', '24', 'F', '172'],
			 ['pietro', '23', 'M', '198'],
			 ['maria', '19', 'F', '175']]
		Y = ['1', '1', '0', '0', '1', '1']
		categorical = [True, False, True, False]
		min_values = [None, None, None, None]
		feature_names = ['name', 'age', 'sex', 'height']
		top=1
		
		# The rule to be discovered should be: IF height in [175, None) -> 1 ELSE 0
		
		print('----------')
		print('LIBRE test')
		print('----------\n')

		print('Training')
		print('---------')
		model = LIBRE()
		model.fit(X, Y, categorical, feature_names, min_values, with_missing_values=False, chi2_threshold=3)
		model.run(heuristic='H1', parallel=True)
		model.simplify(with_filtering=True, min_lsup=0.001, top_K=3, method='WSC', alpha=.7)
		print('> done')
		print()

		print('Testing')
		print('--------')
		predictions, _ = model.predict(X)
		A = model.get_boundary()
		print('Boundary set:', A)
		print('Rule set:')
		model.print_rule_set()
		print('Predictions:', predictions)
		print('> done.')
		print()
		
		print('Testing with a new record')
		print('-------------------------')
		predictions, _ = model.predict([['?', '15', 'M', '151']])
		print('Predictions:', predictions)
		print('> done.')
		print()

		'''
		combined_rule_set = model.combine_boundary_points(model.get_boundary())
		for i in range(0, len(combined_rule_set)):
			rule = combined_rule_set[i]
			print("R" + str(i))
			model.print_rule(rule)
		'''

		'''
		print('List of internal parameters')
		print('----------------------------')
		parameters = model.export_parameters(top)
		for p in parameters:
			print(p, ':', parameters[p])
		print()

		print('Import parameters from dictionary')
		print('----------------------------------')
		model = LIBRE()
		imp_top = model.import_parameters(parameters)
		print('> done')
		print()

		print('Testing with parameters imported from a dictionary')
		print('---------------------------------------------------')
		predictions, _ = model.predict(X, imp_top)
		print('Predictions:', predictions)
		print('> done.')
		print()

		print('Export parameters to json file')
		print('-------------------------------')
		model.export_parameters_to_json('./libre_parameters.json', top)
		print('> done')
		print()

		print('Import parameters from json file')
		print('---------------------------------')
		model = LIBRE()
		imp_top2 = model.import_parameters_from_json('./libre_parameters.json')
		print('> done')
		print()

		print('Testing with parameters imported from a json file')
		print('--------------------------------------------------')
		predictions, _ = model.predict(X, imp_top2)
		print('Predictions:', predictions)
		print('> done.')
		print()
		'''


def main():
	LIBRE.test()

if __name__ == "__main__":
	main()