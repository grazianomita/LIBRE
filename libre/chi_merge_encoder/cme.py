import numpy as np
import sys
import timeit
import pandas as pd
import csv

class ChiMerge:
	'''
	ChiMerge implements the ChiMerge algorithm. ChiMerge receives as input two lists: a list of values and their corresponding labels. Values are first ordered; then, consecutive values - x_i, x_{i+1} - are combined in a discretized range having as extremes the two values, r_i = [x_i, x_{i+1}), only if their combination leads to an interval which purity is above a given threshold. Such a purity measure is 'Chi2' and takes into account the distribution of labels inside the new range. The algorithm ends either when no further combination is possible or as soon as the number of ranges is below a given parameter (max_intervals).
	
	ChiMerge has two main methods:
	
	- fit: It builds the discretized ranges 
	- transform: It discretizes the input values according to the discretized ranges
	'''
	
	def __init__(self, chi2_threshold=4.6, max_intervals=0):
		'''
		It initializes the ChiMerge class. It receives two main parameters.
		
		Parameters:
		------------
			@ "chi2_threshold": Threshold used to merge consecutive intervals
			@ "max_intervals": Max number of allowed intervals
		'''
		self.chi2_threshold = chi2_threshold
		self.max_intervals = max_intervals
		self.__mappings = []
	
	def fit(self, values, labels, number_of_distinct_labels, min_value=None, with_missing_values=False, missing_value_symbol='?'):
		'''
		It runs the chiMerge procedure on a given list of values. At the end of the procedure, we will have an updated chi_merge_table and we will be able to get the discretized ranges from there. It is important to notice that each interval is in the form [left, right).
		
		Parameters:
		------------
			@ "values": List of values to be discretized
			@ "labels": Labels associated to values. They MUST be integers between 0 and K-1, where K is the number of distinct labels
			@ "number_of_distinct_labels": Number of distinct labels
			@ "min_value": min value that can occur in values. It can be None if min_value is -inf. None by default
			@ "with_missing_values": True if "values" may contain missing values
			@ "missing_value_symbol": Symbol used to encode missing values. '?' by default
		'''
		if len(values) != len(labels):
			print("Error: values and labels must have the same size.")
			sys.exit(1)
		
		self.number_of_distinct_labels = number_of_distinct_labels
		self.min_value = min_value
		self.with_missing_values = with_missing_values
		self.missing_value_symbol = missing_value_symbol 
		self.__chi_merge_table = []
		
		# 0. Filter missing values from the list of values.
		if with_missing_values:
			labels = [labels[i] for i in range(0, len(values)) if values[i] != missing_value_symbol]
			values = [values[i] for i in range(0, len(values)) if values[i] != missing_value_symbol]
		
		# 1. Initialize the chi_merge_table.
		self.__initialize_chi_merge_table(values, labels)
		number_of_discretized_ranges = len(self.__chi_merge_table)
		
		# 2. Until end condition is met:
		while (len(self.__chi_merge_table) > self.max_intervals):
			i = 0
			while i < len(self.__chi_merge_table)-1:
				# 2a) Get class frequencies of consecutive ranges r_i, r_{i+1}.
				one = self.__chi_merge_table[i][1]['class_frequency']
				two = self.__chi_merge_table[i+1][1]['class_frequency']
				# 2b) Calculate chi2 between the two ranges.
				chi2_value = self.__chi2(one, two)
				# 2c) Merge the two consecutive records if chi2_value <= chi2_threshold
				if chi2_value <= self.chi2_threshold:
					self.__chi_merge_table[i][1]['chi2'] = chi2_value
					self.__chi_merge_table[i][1]['class_frequency'] = self.__chiMerge(one, two)
					self.__chi_merge_table.pop(i+1)
				i += 1
			# If the number of discretized ranges has not changed -> no further combination is possible -> break.
			if (len(self.__chi_merge_table) == number_of_discretized_ranges):
				break
			number_of_discretized_ranges = len(self.__chi_merge_table)
		
		# 3. Extract mappings to see how values have been mapped to discrete values.
		self.__extract_mappings() # mappings are stored in self.__mappings.
	
	def transform(self, values):
		'''
		It receives as input the list of values to be discretized and returns the corresponding discretized values according to self.__mappings previously computed.
		
		Parameters:
		------------
			@ "values": list of values to be discretized.
		'''
		d_values = []
		for v in values:
			if v == self.missing_value_symbol:
				d_values.append(len(self.__mappings)-1)
				continue
			for i in range(len(self.__mappings)):
				if self.__mappings[i][1] == None or v < self.__mappings[i][1]:
					d_values.append(i)
					break
		return d_values
	
	def __initialize_chi_merge_table(self, values, labels):
		'''
		Given data, where data_i = (value, label), it calculates the class frequency for each distinct value. Class frequencies are stored in a table - chi_merge_table, ordered by value (in ascending order). The ordered chi_merge_table is a list of tuples, where: chi_merge_table_i = (value, {class_frequency=c_f}).
		
		Parameters:
		------------
			@ "values": List of values to be discretized (without missing values)
			@ "labels": Labels associated to values (without missing values)
		'''
		chi_merge_table = {}
		for i in range(len(values)):
			value = values[i]
			label = labels[i]
			if value not in chi_merge_table:
				class_frequency = [0] * self.number_of_distinct_labels
				chi_merge_table[value] = {'class_frequency': class_frequency}
			chi_merge_table[value]['class_frequency'][label] += 1
		self.__chi_merge_table = sorted([[k, v] for k,v in chi_merge_table.items()])
	
	def __chi2(self, one, two):
		'''
		It receives the class frequency of two consecutive intervals (it is a list) and computes the corresponding chi2 value.
		
		Parameters:
		------------
			@ "one": class frequency of interval i in the form [count_C1, ..., count_CK]
			@ "two": class frequency of interval i+1 in the form [count_C1, ..., count_CK]
		'''
		A = [one, two]
		R = [float(sum(one)), float(sum(two))]
		C = [float(one[i] + two[i]) for i in range(0, self.number_of_distinct_labels)]
		N = sum(C)
		chi2_value = .0
		for i in range(2):
			for j in range(self.number_of_distinct_labels):
				E = (R[i] * C[j]) / N
				if E != 0:
					chi2_value = chi2_value + (float(A[i][j] - E)**2 / E)
		return chi2_value
	
	def __chiMerge(self, one, two):
		'''
		It combines the class frequency lists of two consecutive intervals.
		
		Parameters:
		------------
			@ "one": class frequency of interval i in the form [count_C1, ..., count_CK]
			@ "two": class frequency of interval i+1 in the form [count_C1, ..., count_CK]
		'''
		return [one[i]+two[i] for i in range(0, len(one))]
	
	def __extract_mappings(self):
		'''
		Once chiMerge has been executed, the mapping between discrete values and discretized ranges is built. Both min_value and missing values need to be considered in the construction. self.__mappings is a list where mappings_i contains the range of values that have been mapped to the discretized value i: mappings_i = [min, max). It returns True if the operation ends successfully.
		'''
		if len(self.__chi_merge_table) == 0:
			return False
		
		self.__mappings = []
		# 1. First of all, let's check if self.min_values is None or is lower than the lowest value in the data.
		lowest_value_in_data = self.__chi_merge_table[0][0]
		if self.min_value is None or self.min_value < lowest_value_in_data:
			# self.__mappings.append((self.min_value, lowest_value_in_data))
			self.__chi_merge_table[0][0] = self.min_value
		
		# 2. Add the rest of the mappings from chi_merge_table.
		for k in range(0, len(self.__chi_merge_table)-1):
			self.__mappings.append((self.__chi_merge_table[k][0], self.__chi_merge_table[k+1][0]))
		self.__mappings.append((self.__chi_merge_table[len(self.__chi_merge_table)-1][0], None))
		
		# 3. If with_missing_values is True, the mapping for missing_value_symbol is appended at the end.
		if self.with_missing_values:
			 self.__mappings.append((self.missing_value_symbol, self.missing_value_symbol))
		
		return True
	
	def set_parameters(self, chi2_threshold, max_intervals=0):
		'''
		It sets the two main parameters of ChiMerge.
		
		Parameters:
		------------
			@ "chi2_threshold": Threshold used to merge consecutive intervals
			@ "max_intervals": Max number of allowed intervals
		'''
		self.chi2_threshold = chi2_threshold
		self.max_intervals = max_intervals
	
	def get_mappings(self):
		'''
		It returns the mapping between discrete values and discretized ranges.
		'''
		if len(self.__mappings) == 0:
			print("No mapping found. Please, be sure to call 'fit' before calling 'get_mappings'.")
		return self.__mappings

	@staticmethod
	def test(chi2_threshold=2.7, max_intervals=0):
		values = [.5, .5, 91, '?', 30, .9, 9, 85, 85, 41, 91.3, 13.6, 99.1, 98, 92, 41]
		labels = [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 2, 2, 2, 0]
		number_of_distinct_labels = len(set(labels))
		missing_value_symbol = '?'
		with_missing_values = missing_value_symbol in values
		
		print('-------------')
		print('ChiMerge test')
		print('-------------\n')

		print('> Print input values and labels')
		for i in range(len(values)):
			print('\t', values[i], '\t-\t', labels[i])
		print('> Run ChiMerge...')
		chi_merge = ChiMerge(chi2_threshold=chi2_threshold, max_intervals=max_intervals)
		chi_merge.fit(
			values, 
			labels, 
			number_of_distinct_labels, 
			min_value=None,
			with_missing_values=True, 
			missing_value_symbol=missing_value_symbol
		)
		d_values = chi_merge.transform(values)
		print('> Done.')
		print('> List of mappings:')
		for idx, m in enumerate(chi_merge.get_mappings()):
			print('\t', m, '\t-\t', idx)
		
		print('> Discretized values:')
		d_values = chi_merge.transform(values)
		for i in range(len(values)):
			print('\t', values[i], '\t-\t', d_values[i])
		
		print('> Test successfully completed.')


class ChiMergeEncoder:
	'''
	ChiMergeEncoder is used to discretize input datasets. A dataset can have both categorical and numerical features: categorical features are directly mapped to integer values, whereas numerical features are discretized first by using ChiMerge and then converted to integer values. In the current implementation, ChiMerge runs with the same input parameters for all the input features: this may lead to suboptimal results.

	ChiMergeEncoder has two main methods:

	- fit: It builds the mapping between original values and integer values for each feature
	- transform: It converts original input records in discretized records

	P.S. labels must be int or string encoding int!
	'''

	output_format = ['INT', 'OHE_01', 'OHE_BOOL', 'OHE_STR', 'IOHE_STR']

	def fit(self, features, labels, categorical, min_values, with_missing_values=True, missing_value_symbol="?", chi2_threshold=4.6, max_intervals=0):
		'''
		It builds the discretization intervals for each feature.
		
		Parameters:
		------------
			@ "features": List of features in the form [[record_1], ..., [record_N]]
			@ "labels": List of labels. They must encode integers between 0 and K-1, where K is the number of distinct labels
			@ "categorical": List of True/False values. True if the corresponding feature is categorical, False otherwise
			@ "min_values": List of min_values. None if the corresponding features is categorical or the min value is not known
			@ "missing_value_symbol": Symbol used in the dataset to encode missing values
			@ "chi2_threshold": discretization threshold
			@ "max_intervals": max number of allowed intervals
		'''

		if len(features) != len(labels):
			print('> Error: features and labels must have the same length!')
			sys.exit(1)

		features = np.array(features) # features = [[record_1], ..., [record_N]]
		labels = [int(float(l)) for l in labels] # Labels are converted to integers

		self.missing_value_symbol = missing_value_symbol # "?" by default
		self.with_missing_values = with_missing_values # True if it is possible to find missing values in the dataset
		self.__number_of_distinct_labels = len(set(labels)) # Number of distinct labels
		self.__number_of_features = features.shape[1] # Number of features
		self.__categorical = categorical # List of True/False values. True if the corresponding feature is categorical, False otherwise
		self.__feature_domain_sizes = [] # After discretization, each feature will have a discretized domain size
		self.__discretized_ranges = [] # For each feature, it lists the discretized ranges. [[(0,13.1), (13.1, 40)], [(0,.5), (.5,1)]] or [('gra', 'gra'), ('fra', 'fra')]
		self.__categorical_mappings = {} # For categorical features only, it stores the mapping between categorical and discretized value like: ['0': {'gra':0, 'fra':1}, '3':{'a1':0, 'a2':1}]

		if (self.__number_of_features != len(self.__categorical) or self.__number_of_features != len(min_values)):
			print('> Error: the size of categorical and/or min_values is not compatible with the number of features', self.__number_of_features)
			sys.exit(1)
		
		chi_merge = ChiMerge(chi2_threshold=chi2_threshold, max_intervals=max_intervals)
		for i in range(0, self.__number_of_features):
			values = features[:, i]
			if self.__categorical[i] == True:
				discretized_range, mappings = self.__from_categorical_to_int(values, i)
				self.__categorical_mappings[i] = mappings
			else:
				discretized_range = self.__from_numerical_to_int(chi_merge, values, labels, min_value=min_values[i])
			self.__discretized_ranges.append(discretized_range)
			self.__feature_domain_sizes.append(len(discretized_range))

	def transform(self, records, o_format='INT'):
		'''
		It receives a list of records as input and returns a list of discretized records.

		Parameters:
		------------
			@ "records": list of records to be discretized
			@ "o_format": output format. We accept the output format specified in the class variable output_format
		'''

		if len(records) > 0 and len(records[0]) != self.__number_of_features:
			print('> Error: Input record size is wrong!')
			sys.exit(1)

		o_format = o_format.upper()

		if o_format not in ChiMergeEncoder.output_format:
			print('> Error: Output format', o_format, 'not supported!')
			sys.exit(2)

		d_records = []
		for record in records:
			d_record = []
			for i in range(len(record)):
				domain_size = self.__feature_domain_sizes[i]
				value = record[i]
				if self.__categorical[i]:
					d_value = self.__categorical_mappings[i][value]
				else:
					if value == self.missing_value_symbol:
						d_value = domain_size-1
					else:
						value = float(record[i])
						for j in range(len(self.__discretized_ranges[i])):
							if self.__discretized_ranges[i][j][1] == None or value < self.__discretized_ranges[i][j][1]:
								d_value = j
								break
				d_record.append(d_value)
			d_records.append(self.__format(d_record, o_format=o_format))

		return d_records

	def __format(self, d_record, o_format):
		if o_format == 'INT':
			return d_record
		elif o_format == 'OHE_01' or o_format == 'OHE_BOOL':
			if o_format == 'OHE_01':
				pos = [1]
				neg = [0]
			else:
				pos = [True]
				neg = [False]
			output = []
		elif o_format == 'OHE_STR' or o_format == 'IOHE_STR':
			if o_format == 'OHE_STR':
				pos = '1'
				neg = '0'
			else:
				pos = '0'
				neg = '1'
			output = ''
		else:
			return d_record
		for i in range(len(d_record)):
			value = d_record[i]
			for j in range(self.__feature_domain_sizes[i]):
				if j == value:
					output += pos
				else:
					output += neg
		return output

	def __from_categorical_to_int(self, values, index):
		'''
		It converts categorical values to integer values.
		It returns the mappings as a set of key-value pairs and the discretized range with the same structure returned by __from_numerical_to_binary().

		Parameters:
		------------
			@ "values": list of values to convert to integers
			@ "index": id of the currently processed feature 
		'''

		distinct_values = list(np.unique(values)) # Distinct values in alphabetical order
		mappings = {} # "Categorical":int value pairs
		d_r = []
		count = 0
		
		for v in distinct_values:
			mappings[v] = count
			d_r.append((v,v))
			count += 1
		
		if self.with_missing_values:
			if self.missing_value_symbol not in mappings:
				mappings[self.missing_value_symbol] = count
				d_r.append((self.missing_value_symbol, self.missing_value_symbol))
		
		return d_r, mappings

	def __from_numerical_to_int(self, discretizer, values, labels, min_value=None):
		'''
		It converts numerical values to discretized values.
		It returns the set of discretized ranges and the discretized values.
		
		Parameters:
		------------
			@ "discretizer": discretizer used to discretize values: ChiMerge
			@ "values": values to be discretized - corresponding to a given feature
			@ "labels": labels corresponding to values to be discretized
			@ "min_value": min_value that the current feature can take
		'''

		# Numeric values are converted to float (except the missing value symbol that is encoded with self.missing_value_symbol)
		if self.with_missing_values:
			values = [float(v) if v != self.missing_value_symbol else v for v in values]
		else:
			values = [float(v) for v in values]

		# Run the discretizer
		discretizer.fit(values, labels, self.__number_of_distinct_labels, min_value=min_value, with_missing_values=self.with_missing_values, missing_value_symbol=self.missing_value_symbol)
		discretized_range = discretizer.get_mappings()
		
		return discretized_range

	def get_discretized_ranges(self):
		'''
		It returns the list of discretized ranges for each feature.
		'''
		return self.__discretized_ranges

	def get_discrete_sizes(self):
		'''
		It returns the list of discretized sizes, one value per feature.
		'''
		return self.__feature_domain_sizes

	def import_parameters(self, parameters):
		'''
		It receives a dictionary with the following format.
		This is just an example.
		
		parameters = {
			'with_missing_values':False,
			'missing_value_symbol':'?',
			'number_of_distinct_labels':2,
			'number_of_features':3,
			'categorical':[True, True, False],
			'feature_domain_sizes':[2, 2, 3],
			'discretized_ranges':[[('gra', 'gra'), ('fra', 'fra')], [('M', 'M'), ('F', 'F')], [(0, 150), (150, 170), (170, 210)]]
			'categorical_mappings':['0': {'gra':0, 'fra':1}, '1':{'M':0, 'F':1}]
		}
		'''
		keywords = ['with_missing_values', 'missing_value_symbol', 'number_of_distinct_labels', 'number_of_features', 'categorical', 'feature_domain_sizes', 'discretized_ranges', 'categorical_mappings']

		for key in keywords:
			if key not in parameters:
				print('> Error: Parameter', key, 'is missing')
				sys.exit(3)

		self.with_missing_values = parameters['with_missing_values']
		self.missing_value_symbol = parameters['missing_value_symbol']
		self.__number_of_distinct_labels = parameters['number_of_distinct_labels']
		self.__number_of_features = parameters['number_of_features']
		self.__categorical = parameters['categorical']
		self.__feature_domain_sizes = parameters['feature_domain_sizes']
		self.__discretized_ranges = parameters['discretized_ranges']
		# When 'parameters' is imported from a json file, all keys are string!
		# Therefore we need to force type casting.
		self.__categorical_mappings = {}
		for key in parameters['categorical_mappings']:
			self.__categorical_mappings[int(key)] = parameters['categorical_mappings'][key]

	def export_parameters(self):
		'''
		It export all parameters needed to discretize an input dataset.
		'''
		parameters = {}
		parameters['with_missing_values'] = self.with_missing_values
		parameters['missing_value_symbol'] = self.missing_value_symbol
		parameters['number_of_distinct_labels'] = self.__number_of_distinct_labels
		parameters['number_of_features'] = self.__number_of_features
		parameters['categorical'] = self.__categorical
		parameters['feature_domain_sizes'] = self.__feature_domain_sizes
		parameters['discretized_ranges'] = self.__discretized_ranges
		parameters['categorical_mappings'] = self.__categorical_mappings
		return parameters

	def import_parameters_from_json(self, filename):
		with open(filename, 'r') as infile:
			parameters = json.load(infile)
		self.import_parameters(parameters)

	def export_parameters_to_json(self, filename):
		parameters = self.export_parameters()
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
		Y = ['1', '1', '0', '0', '1', '0']
		categorical = [True, False, True, False]
		min_values = [None, None, None, None]
		feature_names = ['name', 'age', 'sex', 'height']
		missing_value_symbol = '?'
		
		print('--------------------')
		print('ChiMergeEncoder test')
		print('--------------------\n')
		
		print('> Print input values and labels')
		for i in range(len(X)):
			print('\t', X[i], '-', Y[i])
		
		print('> Run ChiMergeEncoder...')
		encoder = ChiMergeEncoder()
		encoder.fit(X, Y, categorical, min_values, with_missing_values=True, missing_value_symbol=missing_value_symbol, chi2_threshold=1)
		print('> Done.')
		print('> Discretized ranges per feature:')
		for i in range(len(feature_names)):
			print('\t', feature_names[i], ':', encoder.get_discretized_ranges()[i])
			assert isinstance(encoder.get_discretized_ranges()[i], list)
		
		print('> Discretized records (INT format):')
		X_ = encoder.transform(X, o_format='INT')
		for x_ in X_:
			print('\t', x_)
			
		print('> Discretized records (OHE_STR format):')
		X_ = encoder.transform(X, o_format='OHE_STR')
		for x_ in X_:
			print('\t', x_)
		
		print('> Discretized records (IOHE_STR format):')
		X_ = encoder.transform(X, o_format='IOHE_STR')
		for x_ in X_:
			print('\t', x_)

		print('> Test successfully completed.')


def main():
	ChiMergeEncoder.test()
	# ChiMerge.test()

if __name__ == "__main__":
	main()