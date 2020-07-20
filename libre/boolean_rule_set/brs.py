import sys
import timeit
import numpy as np
import multiprocessing
from multiprocessing import Pool
from collections import Counter

def unwrap_self_run(arg, **kwarg):
	return BooleanRuleSet.run(*arg, **kwarg)
	
def unwrap_self_compute_boundary_statistics(arg, **kwarg):
	return BooleanRuleSet.compute_boundary_statistics(*arg, **kwarg)

class BooleanRuleSet:
	'''
	BooleanRuleSet (BRS) is an approximate algorithm that lears a set of boundary points A from T and F, where T and F are two posets of positive and negative records, respectively. The algorithm considers one positive record at a time and generalizes it being sure to avoid conflicts with F. BRS can eventually run in parallel on multiple processes. It is highly reccomended to run BRS in parallel to boost execution time. BRS is inspired by Muselli,	2003: "Shadow Clustering: A Method for Monotone Boolean Function Synthesis".
	For more details refer to section 3 in Mita, 2020 - "LIBRE: Learning Interpretable Boolean Rule Ensembles".
	'''
	
	def __init__(self, heuristic='H1', parallel='True'):
		'''
		It sets the main parameters.
		
		Parameters:
		------------
			@ "heuristic": search heuristic
			@ "parallel": True by default
		'''
		self.set_heuristic(heuristic)
		self.set_parallel(parallel)
		self.set_T([]) # It also computes self.positive_number_of_records and self.__T_occurrences
		self.set_F([]) # It also computes self.negative_number_of_records and self.__F_occurrences
		self.set_boundary([]) # It sets the boundary and simplified boundary to an empty set
		self.__boundary_statistics = {} # For each element a of the boundary, it stores: [cov_T, cov_F, num_zeros].
		self.__domain_sizes = [] # Number of bits associated to input features. Used to compute boundary statistics
		self.__number_of_records = self.number_of_positive_records + self.number_of_negative_records
		self.__record_size = 0 # Initialize record_size
	
	def fit(self, T, F, domain_sizes):
		'''
		BRS generates a boundary set for the pair (T, F).
		
		Parameters:
		------------
			@ "T": set of binary records (strings) with label 1
			@ "F": set of binary records (strings) with label 0
		'''
		if len(T) == 0:
			print('Error: T must have at least one element.')
			sys.exit(1)
		if len(F) > 0:
			if len(T[0]) != len(F[0]):
				print('Error: T and F must have the same record size.')
		
		self.set_T(T)	# It sets self.__T, self.number_of_positive_records, self.__T_occurrences.
		self.set_F(F)	# It sets self.__F, self.number_of_negative_records, self.__F_occurrences.
		self.set_boundary([]) # At the beginning, self.__boundary = [].
		self.__compute_reference_indexes(T, F) # It sets self.____T_reference_indexes, self.__F_reference_indexes, self.__T_occurrences, self.__F_occurrences.
		self.__boundary_statistics = {} # For each element of the boundary, it stores: [cov_T, cov_F, num_of_zeros].
		self.__domain_sizes = domain_sizes # Used indirectly to compute lsup and escl later in simplify. [10, 2].
		self.__number_of_records = self.number_of_positive_records + self.number_of_negative_records
		self.__record_size = len(T[0])
	
		# 1) If heuristic is H1 or H2, we initialize some structures used by the greedy procedures.
		self.__initialize_search_parameters()
	
		# 2) If "parallel" is True, multiple processes are spawned.
		self.__boundary, self.__boundary_statistics = self.__find_boundary()
		self.__simplified_boundary = list(self.__boundary)	# At this point, boundary and simplified boundary are the same.

	def run(self, params):
		'''
		It executes the search algorithm on a subset of the positive input records.
	
		Parameters:
		------------
			@ "params": [start, end]. It defines the range of records in T to be processed
		'''
		# 1) Set A = 0 and S = T.
		S = list(self.__T[params[0]:params[1]])
		A = [] # Modified by reference in self.__GSC().
		boundary_statistics = {}
		
		# 2) While S is not empty.
		count = 1
		while len(S) > 0:
			# 2a) Choose x in S (first element in our case).
			x = S[0]
			I = self.__get_I_from_x(x) # I contains the list of candidate positive bits to flip-off.
			J = [] # It stores the indexes of bits that cannot be flip-off.
			# 2b) Add to A one or more boundary points for the pair (T, F) -> Run BRS.
			GSC_dict = {}
			for i in I:
				GSC_dict[i] = {}
				GSC_dict[i]['T0_cardinality'] = self.__T0_cardinality[i]
				GSC_dict[i]['S0_cardinality'] = self.__S0_cardinality[i]
				GSC_dict[i]['T0'] = self.__T0[i]
				GSC_dict[i]['F0'] = self.__F0[i]
				GSC_dict[i]['distance'] = self.__distance(x, self.__F0[i])
			a, cov_T, cov_F, num_of_zeros = self.__GSC(A, I, J, GSC_dict)
			if a != None:
				boundary_statistics[a] = [cov_T, cov_F, num_of_zeros]
			# 2c) Remove x from S and update S0_cardinality.
			for k in range(0, len(x)):
				if x[k] == 0:
					GSC_dict[k]['S0_cardinality'] -= 1
			# In theory we should eliminate not only x but also all elements in S that cover a and update ['S0_cardinality'] accordingly. However, this is not done to increase the probability of retrieving different boundary points.
			S.remove(x)
			'''
			elements_to_be_removed = {s for s in S for a in A if self.__covers(a, s)}
			for e in elements_to_be_removed:
			    if flag == 'H1' or flag == 'H2':
			        for i in range(0, len(e)):
			            if e[i] == 0:
			                S0_cardinality[i] -= 1
			        S.remove(e)
			'''
			count += 1
		return boundary_statistics
	
	def __compute_reference_indexes(self, T, F):
		'''
		It finds the correspondence between elements in T/F (that might contain duplicates) and the corresponding index. BRS works with sets! It is true that we store a count of each element, but this is not enough! Every time we calculate the elements covered by a given boundary point, we also need to know which records in the original T it covers. Only in this way we can be sure to apply weighted_set_cover (and its parameters) correctly.
		
		Parameters:
		------------
			@ "T": Set of positive binary records (strings)
			@ "F": Set of negative binary records (strings)
		'''
		self.__T_reference_indexes = {}
		self.__F_reference_indexes = {}
		for idx, t in enumerate(T):
			if t not in self.__T_reference_indexes:
				self.__T_reference_indexes[t] = [idx]
			else:
				self.__T_reference_indexes[t].append(idx)
		for idx, f in enumerate(F):
			if f not in self.__F_reference_indexes:
				self.__F_reference_indexes[f] = [idx]
			else:
				self.__F_reference_indexes[f].append(idx)
		
	def __initialize_search_parameters(self):
		'''
		It initializes some parameters used during the research of boundary points.
		'''
		if self.heuristic.upper() in ['H1', 'H2']:
			self.__T0_cardinality = [] # For each i, it stores the number of records in T having T[i] = 0
			self.__S0_cardinality = [] # For each i, it stores the number of records in S having S[i] = 0
			self.__T0 = [] # For each i, it stores the list of records id (indexes) in T having T[i] = 0
			self.__F0 = [] # For each i, it stores the list of records in F having F[i] = 0
			for i in range(0, len(self.__T[0])):
				self.__T0_cardinality.append(sum([1 for t in self.__T if t[i] == '0']))
				self.__T0.append([idx for idx, t in enumerate(self.__T) if t[i] == '0'])
				self.__F0.append([idx for idx, f in enumerate(self.__F) if f[i] == '0'])
			self.__S0_cardinality = list(self.__T0_cardinality)

	def __find_boundary(self):
		'''
		It executes self.run() either in parallel or sequentially to find the list of boundary points, together with their statistics.
		'''
		if self.parallel:
			processes = multiprocessing.cpu_count() if len(self.__T) > multiprocessing.cpu_count() else len(self.__T)
			params = self.__extract_index_ranges_for_parallel_run(len(self.__T), processes)
			if __name__ == "boolean_rule_set.brs":
				pool = Pool(processes=processes)
				pool_outputs = pool.map(unwrap_self_run, zip([self]*len(params), params))
				pool.close()
				pool.join()
			# Each process returns a dictionary with boundary point as key and some statistics as value: [pos_T, pos_F, num_zeros].
			boundary_statistics_tmp = {boundary_point:p[boundary_point] for p in pool_outputs for boundary_point in p}
		else:
			boundary_statistics_tmp = self.run([0, len(self.__T)])

		# Boundary points are combined by ensuring that there are no elements that cover other ones.
		boundary = []
		boundary_statistics = {}
		for bp in boundary_statistics_tmp:
			is_covered = False
			for a in boundary:
				if self.__covers(a, bp) == True:
					is_covered = True
					break
			if is_covered == False:
				boundary.append(bp)
				boundary_statistics[bp] = boundary_statistics_tmp[bp]

		return boundary, boundary_statistics

	def __GSC(self, A, I, J, GSC_dict):
		'''
		Greedy procedure to look for boundary points. It might run H1 or H2 depending on self.heuristic.
		The only difference between the two heuristics is the greedy procedure implemented inside 
		self.__get_best_i.
	
		Parameters:
		------------
			@ "A": Boundary set as reference
			@ "I": List of indexes equal to 1 in the selected string x in S
			@ "J": List containing the indexes that cannot be flip-off
			@ "GSC_dict": dictionary containing |T0_i|, |S0_i|, F0 and distance for all i in I
		'''
		# 1) While I is not empty
		while (len(I) > 0):
			# 1a) Move from I to J all the elements i having the distance between p(I U J) and F0_i = 1.
			# p_I_U_J = self.__get_x_from_I(I+J)
			i_to_be_moved_to_J = []
			for i in I:
				if GSC_dict[i]['distance'][0][0] == 1:
					i_to_be_moved_to_J.append(i)
			for i in i_to_be_moved_to_J:
				I.remove(i)
				J.append(i)
			if len(I) == 0:
				break
			# 1b) Choose the best i to be flip-off.
			best_i = self.__get_best_i(GSC_dict, I)
			# 1c) Remove i from I and update the distance.
			I.remove(best_i)
			for i in I:
				self.__update_distance(GSC_dict[i]['F0'], best_i, GSC_dict[i]['distance'])
	
		# 2) If there is no a in A such that p(J) covers a, p(J) is added to A.
		p_J = self.__get_x_from_I(J)
		# 2a) If we obtain an empty string, we return immediately, no information has been learned.
		if(int(p_J) == 0):
			return None, None, None, None
		# 2b) Check for redundancy
		is_covered = False
		for a in A:
			if self.__covers(a, p_J) == True:
				is_covered = True
				break
		if is_covered == False:
			A.append(p_J)
			# Only if the boundary point is going to be added to the boundary set, we calculate the boundary point statistics.
			covT, covF, num_of_zeros = self.__compute_boundary_point_statistics(p_J)
			return p_J, covT, covF, num_of_zeros

		return None, None, None, None

	def __compute_boundary_point_statistics(self, bp):
		'''
		Given a boundary point, it computes the boundary point statistics: covT, covF, num_of_zeros, where covT and covF is the set of positive and negative covered samples respectively.
		
		Parameters:
		------------
			@ "bp": boundary point
		'''
		num_of_zeros = 0
		# When we generate a boundary point, we also need to take track of which positive and negative samples it covers. This information is stored in cov_pos_grouped_by_features and cov_neg_grouped_by_features, respectively.
		# If we do the intersections of covered elements among features in cov_pos_grouped_by_features/cov_neg_grouped_by_features, we get the number of cov_pos_samples and cov_neg_samples.
		cov_pos_grouped_by_features = [set() for _ in range(len(self.__domain_sizes))] # for each feature, it contains the covered pos samples. For example: [set('110'), set('101', '011')].
		cov_neg_grouped_by_features = [set() for _ in range(len(self.__domain_sizes))] # for each feature, it contains the covered neg samples. For example: [set('110'), set('101', '011')].
		domain_sizes_extremes = np.cumsum(self.__domain_sizes)

		for idx, bit in enumerate(bp):
			if bit == '0':
				num_of_zeros += 1
				for k in range(len(domain_sizes_extremes)):
					if idx < domain_sizes_extremes[k]:
						break
				cov_pos_grouped_by_features[k].update(self.__T0[idx])
				cov_neg_grouped_by_features[k].update(self.__F0[idx])

		full_domain_cov_pos_grouped_by_features = [set() for _ in range(len(self.__domain_sizes))]
		for idx, element in enumerate(cov_pos_grouped_by_features):
			for e in element:
				full_domain_cov_pos_grouped_by_features[idx].update(self.__T_reference_indexes[self.__T[e]])

		full_domain_cov_neg_grouped_by_features = [set() for _ in range(len(self.__domain_sizes))]
		for idx, element in enumerate(cov_neg_grouped_by_features):
			for e in element:
				full_domain_cov_neg_grouped_by_features[idx].update(self.__F_reference_indexes[self.__F[e]])

		if len(domain_sizes_extremes) == 1:
			return full_domain_cov_pos_grouped_by_features[0], full_domain_cov_neg_grouped_by_features[0], num_of_zeros
		else:
			return full_domain_cov_pos_grouped_by_features[0].intersection(*full_domain_cov_pos_grouped_by_features[1:]), full_domain_cov_neg_grouped_by_features[0].intersection(*full_domain_cov_neg_grouped_by_features[1:]), num_of_zeros

	def __get_best_i(self, GSC_dict, I):
		'''
		It chooses the best index i in I to flip-off at a given iteration.
	
		Parameters:
		------------
			@ "GSC_dict": dictionary containing |T0_i|, |S0_i|, F0, and distance for all i in I
			@ "I": List of candidates to flip-off
		'''
		best_i = I[0]
		if self.heuristic.upper() == 'H1':
			for i in I:
				if GSC_dict[i]['S0_cardinality'] > GSC_dict[best_i]['S0_cardinality']:
					best_i = i
				elif GSC_dict[i]['S0_cardinality'] == GSC_dict[best_i]['S0_cardinality']:
					if GSC_dict[i]['T0_cardinality'] > GSC_dict[best_i]['T0_cardinality']:
						best_i = i
					elif GSC_dict[i]['T0_cardinality'] == GSC_dict[best_i]['T0_cardinality']:
						if GSC_dict[i]['distance'][0] > GSC_dict[best_i]['distance'][0]:
							best_i = i
		elif self.heuristic.upper() == 'H2':
			for i in I:
				if GSC_dict[i]['distance'][0] > GSC_dict[best_i]['distance'][0]:
					best_i = i
				elif GSC_dict[i]['distance'][0] == GSC_dict[best_i]['distance'][0]:
					if GSC_dict[i]['S0_cardinality'] > GSC_dict[best_i]['S0_cardinality']:
						best_i = i
					elif GSC_dict[i]['S0_cardinality'] == GSC_dict[best_i]['S0_cardinality']:
						if GSC_dict[i]['T0_cardinality'] > GSC_dict[best_i]['T0_cardinality']:
							best_i = i
		
		return best_i
	
	def __distance(self, x, Y):
		'''
		ld(x,y) = SUM_i |x_i - y_i|+ where |z|+ = z if z>= 0, 0 otherwise. ld(x,Y) = min(ld(x,y))
	
		Parameters:
		------------
			@ "x": binary string
			@ "Y": list of binary indexes corresponding to elements in self.__F
		'''
		ld = []
		for k in Y:
			y = self.__F[k]
			v = sum([1 for i in range(0, len(y)) if x[i] == '1' and y[i] == '0'])
			ld.append(v)
			if v == 1:
				return [[1], ld]
		if len(ld) == 0:
			return [[-1], ld]
		return [[min(ld)], ld]
	
	def __update_distance(self, Y, best_i, GSC_dict_i_distance):
		'''
		For each positive record, the distance is computed only once. Then, every time an index i in I is chosen as best index, we simply calculate its effect on the distance.
		
		Parameters:
		------------
			@ "Y": list of binary strings in F0[i]
			@ "best_i": index chosen by the greedy procedure
			@ GSC_dict_i_distance: structure with the distance and list of distances
		'''
		# for each element in F0[i]
		if len(Y) == 0:
			return
		for k in range(len(Y)):
			y = self.__F[k]
			# check if y[best_i] is 0. If so, we need to decrease the distance
			# corresponding to F0[i][k].
			if y[best_i] == '0':
				GSC_dict_i_distance[1][k] -= 1
			if GSC_dict_i_distance[1][k] == 1:
				GSC_dict_i_distance[0][0] = 1
				return
		GSC_dict_i_distance[0][0] = min(GSC_dict_i_distance[1])
	
	def __get_I_from_x(self, x):
		'''
		Given a binary string, it returns the list of indexes that are equal to one.
	
		Parameters:
		------------
			@ x: binary string
		'''
		return [i for i in range(0, len(x)) if x[i] == '1']
	
	def __get_x_from_I(self, I):
		'''
		Given a list of indexes, it returns the corresponding binary string.
	
		Parameters:
		------------
			@ I: list containing the indexes that are equal to 1
		'''
		tmp = ['1' if i in I else '0' for i in range(0, self.__record_size)]
		return ''.join(map(str, tmp))
	
	def __covers(self, x, y):
		'''
		It checks if x covers y (x <= y).
		x covers y if x OR y = y.
	
		Parameters:
		------------
			@ x: binary string
			@ y: binary string
		'''
		if len(x) != len(y):
			raise ValueError('The two binary strings have different size.')
		for i in range(0, len(x)):
			xi_or_yi = '0'
			if x[i] == '1' or y[i] == '1':
				xi_or_yi = '1'
			if xi_or_yi != y[i]:
				return False
		return True
	
	def __count_occurrences(self, X):
		'''
		It returns a counter object (dict) counting the occurrences of each element in X.

		Parameters:
		------------
			@ "X": list of samples to which we want to apply Counter()
		'''
		return Counter(X)
	
	def __extract_index_ranges_for_parallel_run(self, N, processes):
		'''
		It generates a list of indexes referred to samples to be assigned to each process.

		Parameters:
		------------
			@ "N": number of samples
			@ "processes" number of processes
		'''
		if processes > N:
			processes = N
		shift = int(N/processes) # Number of elements processed by each process.
		start = 0
		end = shift
		params = []
		for i in range(processes-1):
			params.append([start, end])
			start = end
			end = start + shift		
		params.append([start, N])
		return params

	def get_heuristic(self):
		'''
		It returns self.heuristic.
		'''
		return self.heuristic

	def set_heuristic(self, heuristic):
		'''
		It set self.heuristic. If an invalid parameter is passed, H1 is automatically set.

		Parameters:
		------------
			@ "heuristic": used to set the specific version on BRS
		'''
		if heuristic.upper() in ['H1', 'H2']:
			self.heuristic = heuristic.upper()
		else:
			self.heuristic = 'H1'

	def is_parallel(self):
		'''
		True if BRS runs in parallel.
		'''
		return self.parallel

	def set_parallel(self, parallel=True):
		'''
		It sets the field parallel.

		Parameters:
		------------
			@ "parallel": True if BRS runs in parallel
		'''
		self.parallel = parallel

	def get_T(self):
		'''
		It returns the positive set T. It can return an empty element.
		'''
		return self.__T

	def set_T(self, T):
		'''
		It sets self.__T = T. It also computes the number of positive records, occurrences for each record, and removes duplicates in T.

		Parameters:
		------------
			@ "T": Set of positive binary records (strings)
		'''
		self.__T = T
		self.number_of_positive_records = len(T)
		self.__T_occurrences = self.__count_occurrences(T) # count number of occurrences of each binary record in T.
		self.__simplify_T()

	def __simplify_T(self):
	
		'''
		Given a couple of element x, y in T, if y >= x (x is covered by y), y can be removed from T. In reality, if we use inverse one hot encoding (as we do), there is no need for simplification: we simply remove duplicated elements.
		'''
		self.__T = list(set(self.__T))

	def get_F(self):
		'''
		It returns the negative set F. It can return an empty element.
		'''
		return self.__F

	def set_F(self, F):
		'''
		It sets self.__F = F. It also computes the number of negative records, occurrences for each record, and removes duplicates from F.

		Parameters:
		------------
			@ "F": Set of negative binary records (strings)
		'''
		self.__F = F
		self.number_of_negative_records = len(F)
		self.__F_occurrences = self.__count_occurrences(F) # count number of occurrences of each binary record in F
		self.__simplify_F()

	def __simplify_F(self):
		'''
		Given a couple of element x, y in F, if y >= x (x is covered by y), x can be removed from F. In reality, if we use inverse one hot encoding (as we do), there is no need for simplification: we simply remove duplicated elements.
		'''
		self.__F = list(set(self.__F))

	def get_boundary(self):
		'''
		It returns the (simplified) boundary set learned by BRS.
		'''
		return list(self.__simplified_boundary)

	def set_boundary(self, A):
		'''
		It sets the boundary to be equal to A.

		Parameters:
		------------
			@ "A": Boundary
		'''
		self.__boundary = A
		self.__simplified_boundary = A
	
	def get_boundary_statistics(self):
		'''
		It returns the boundary statistics, a dictionary {bp : [cov_T, cov_F, num_zeros]}.
		'''
		if self.__boundary_statistics == []:
			print("boundary_statistics is empty. Please run fit() first!")
			sys.exit(1)

		return self.__boundary_statistics

	def set_boundary_statistics(self, A_statistics):
		'''
		It sets the boundary statistics.

		Parameters:
		------------
			@ "A_statistics": One record per a in A -> {a : [cov_T, cov_F, num_zeros]}
		'''
		self.__boundary_statistics = A_statistics