#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import os
import time
import glob
from scipy.cluster.hierarchy import linkage,fcluster
from sklearn import linear_model
import multiprocessing
from scipy.spatial.distance import cdist, pdist
import math
from data_loader import *
from save_result import *

#***********************************CODE USAGE GUIDE*********************************************
#                             		Work for FSE 2018 
#
# cascading_clustering.py takes the following data files as input:
# 1. Log sequence matrix files, each file contains a matrix of many log sequence vectors, which
#    represents the occurrences of log events generated in the corresponding time interval. 
#    Note that duplicates would be removed automatically by our code.
#
# 2. KPI list, one KPI per time interval, the list is saved in KPI.csv
#
# Parameters are set in the "para" of run.py
#************************************************************************************************


def get_corr_weight(eveOccuMat, kpiList):
	""" Calculate the correlation weight of each log event via linear regression

	Args:
	--------
	eveOccuMat: number of sequences that contain each event, obtained from loading_all_data()
	kpiList: the list of KPIs, obtained from load_kpi()

	Returns:
	--------
	corWeightList: the correlation weight list
	"""
	print('KPI weighting calculation...')
	kpiLists = np.array(kpiList.flatten())
	numInst, numEve = eveOccuMat.shape
	print("event occurrance matrix is of size (%d, %d)"%(numInst, numEve))

	#use linear regression with L2 Norm to calculate the correlation weights.
	reg = linear_model.Ridge(alpha = 0.01, tol = 0, max_iter= 1000000)
	kpiLists = np.expand_dims(kpiLists, axis=1)
	reg.fit(eveOccuMat, kpiLists)
	coefList = reg.coef_[0]

	#in case that coefficient is negative
	corWeightList = [x if x>0 else 0.00001 for x in coefList]
	return corWeightList


def weighting(allrawData, corWeightList):
	""" weighting the data with weights, important events are given more weights.
	Args:
	--------
	allrawData: the big matrix of all log sequence vectors, obtained from loading_all_data()
	corWeightList: correlation weights list, obtained from get_corr_weight()

	Returns:
	--------
	weightedData: weighting the log sequence matrix with IDF weights and correlation weights,
					  as described in Section 3.2
	finalweightList: the final weights list, which combines IDF weights and correlation weights.
	"""

	t0 = time.time()
	numInst, numEvents = allrawData.shape

	# IDF weights calculation
	weightList = []
	for j in range(numEvents):
		cnt = np.count_nonzero(allrawData[:, j])
		weightList.append(math.log((numInst + 1) / (float(cnt) + 1)))
	weightList -= np.mean(weightList)
	newweightList = np.array([1 / float(1 + np.exp(- x)) for x in weightList])

	# combine IDF weight and correlation weight,
	alpha = 0.8
	beta = 0.2
	finalweightList =  beta * newweightList + alpha * np.array(corWeightList)

	# weight the data with final weights.
	weightedData = np.multiply(allrawData, finalweightList)
	print('Step 2. Data Weighting, the weighted data size is %d, it takes %f'%(weightedData.shape[0], time.time()- t0))
	return weightedData, finalweightList


def sampling(input_data, sample_rate):
	""" do randomly sampling from a large input_data with given sample_rate.

	Args:
	--------
	input_data: input large data matrix to be sampled.
	sample_rate: sample percentage, integer number, e.g., 100 represents one is selected out of 100 data instances.

	Returns:
	--------
	sample_data: the sampled data
	"""

	sample_data = []
	for i, line in enumerate(input_data):
		if i%sample_rate == 0:
			sample_data.append(line)
	sample_data = np.array(sample_data)
	print('Step 3. Sampling with sample_rate %d, the original data size is %d, after sampling, the data size is %d' %(sample_rate, input_data.shape[0], sample_data.shape[0]))
	return sample_data


def clustering(para, data):
	""" cluster log sequence vectors into various clusters.

	Args:
	--------
	para: the dictionary of parameters, set in run.py
	data: the data matrix used for clustering

	Returns:
	--------
	seq_clusters: list of lists, data of each cluster is stored in one list, various clusters composes a large list
	"""

	# calculate the distance between any two vectors
	print('Step 4. Distance Calculation: start building distance matrix')
	t_disMat = time.time()
	data_dist = dist_compute(data)
	print("------Distance Calculation finished, it takes %.15f seconds"%(time.time() - t_disMat))

	# check whether the number of distances is correct
	instNum = data.shape[0]
	if len(data_dist) != (instNum-1)*instNum/2:
		print('Error distance matrix size, recomputing')
		data_dist = dist_compute(data)

	# special case handling: if only one vector in the data, no need to do clustering, directly return the data.
	if instNum == 1:
		return [[data[0]]]

	# use hierarchical clustering
	print('Step 5. Clustering, start hierarchical clustering')
	Z = linkage(data_dist, 'complete')
	clusterLabels = fcluster(Z, para['thre'], criterion='distance')
	clusNum = len(set(clusterLabels))
	print('------there are altogether ## %d ## clusters in current clustering' % (clusNum))

	# get cluster list
	seq_clusters = [[] for _ in range(clusNum)]
	for i, ci in enumerate(clusterLabels):
		seq_clusters[ci-1].append(data[i])
	return seq_clusters


def repres_extracting(seq_clusters):
	""" extract the representative vector for each cluster of data.

	Args:
	--------
	seq_clusters: list of clusters of sequence data

	Returns:
	--------
	repre_seqs: list of representatives
	"""
	repre_seqs = []
	for clu in seq_clusters:
		repre_seqs.append(np.mean(clu, axis = 0))
	repre_seqs = np.array(repre_seqs)
	return repre_seqs


def matching(para, weight_data, repre_seqs, curfileIndex, raw_index, raw_data):
	""" match all weighted data (1st round) or mismatched data (other rounds) with cluster representatives.

	Args:
	--------
	para: the dictionary of parameters, set in run.py
	weight_data: weighted sequence data, can be all weighted data (1st round) or mismatched data (other rounds).
	repre_seqs: list of extracted representative per cluster.
	curfileIndex: index used for file naming when saving each cluster into a file
	raw_index: store the sequence index in the raw data, used when saving cluster into files, obtained in loading_all_data()
	raw_data: unweighted raw data. it is used for saving into files, raw data are saved without weighting.

	Returns:
	--------
	mismatch_index: index for mismatched data
	mismatch_data: mismatched data
	curfileIndex: updated curfileIndex
	new_raw_index: updated raw_index where the matched index are removed.
	clu_result: the obtained cluster for each sequence data
	"""

	t1 = time.time()
	print("Step 6. Matching, start matching with original data", weight_data.shape)

	# calculate the distances among weighted data and representative list, find the neareset cluster for each sequence data
	disMatr = cdist(weight_data, repre_seqs)
	minIndexList = np.argmin(disMatr, axis = 1)
	validMinList = disMatr[np.arange(len(minIndexList)), minIndexList] < para['thre']
	clu_array = np.array([minIndexList[i] if x else -1 for i, x in enumerate(validMinList)])
	clu_result = np.array(np.vstack([raw_index, clu_array]).T)

	# get the mismatched data with its index
	if -1 in clu_array:
		mismatch_index = np.where(clu_array == -1)[0]  # mismatched sequence indexes
	else:
		mismatch_index = []
	new_raw_index = [raw_index[i] for i in mismatch_index]
	mismatch_data = np.array([weight_data[i] for i in mismatch_index])
	t2 = time.time()
	print('------matching takes %.15f seconds, %d sequences are not matched' % ( t2- t1, len(mismatch_index)))

	# choose whether to save the matched clusters into files, set in run.py. Although multiprocessing is applied,
	# This costs a lot, turn it off if not needed. False by default
	if para['saveFile']:
		curfileIndex = saveMatching(para, raw_data, clu_array, curfileIndex, raw_index)
		print('------saving matched files takes %.15f seconds' % ( time.time()- t1))
	return mismatch_index, mismatch_data, curfileIndex, new_raw_index, clu_result


def cascading(para, raw_data, rawIndex , weight_data):
	""" the main function of cascading clustering, the iterative process is defined here.

	Args:
	--------
	para: the dictionary of parameters, set in run.py
	raw_data: unweighted raw data. it is used for saving into files, raw data are saved without weighting.
	weight_data: weighted sequence data, can be all weighted data (1st round) or mismatched data (other rounds).
	rawIndex: store the sequence index in the raw data, used when saving cluster into files, obtained in loading_all_data()

	Returns:
	--------
	finaclu_result: the final clustering results
	"""

	# initialize some parameters and variables, get the saving folder ready if needed.
	sample_rate = para['sample_rate']
	cascNum = 100 #maximum number of cascading rounds, user can early stop the process if setting a small value, process all data if it is a large value
	round = 0  #current round of iterative process
	curfileIndex = 0  #current file index, used for saving
	all_repres = [] #store all representatives
	path = para['output_path']
	if not os.path.exists(path):
		os.makedirs(path)
	else:
		deleteAllFiles(path)
	overclu_res = [] # temp used to save clustering results

	# start cascading clustering, sampling, clustering, matching.
	while round < cascNum:
		print('==========round %d========'%round)
		# Sampling Step
		# if mismatched data size is small(e.g., 1000), directly clustering without sampling.
		if weight_data.shape[0]>= 1000:
			sample_weight_data = sampling(weight_data, sample_rate)
		else:
			sample_weight_data = weight_data

		# if the sampled size is <= 1, directly clustering the original data, then the mismatched would be of size 0,
		if  sample_weight_data.shape[0] <= 1:
			sample_weight_data = weight_data

		# Clustering Step, and extract representatives and add into all_repres
		t1 = time.time()
		seq_clusters = clustering(para, sample_weight_data)
		print('----the total time for distance calculation and clustering is', time.time() - t1)
		repre_seqs = repres_extracting(seq_clusters)
		all_repres.extend(repre_seqs)

		# Matching Step, save the clustering results
		mismatch_index, mismatch_data, curfileIndex, rawIndex, clu_result= matching(para, weight_data, repre_seqs, curfileIndex, rawIndex, raw_data)
		overclu_res.append(clu_result)

		# Mismatched data will be processed again.
		weight_data = mismatch_data
		final_remain_index = rawIndex
		if mismatch_data.shape[0] == 0:
			print('cascading stopped as no data left as mismatched.')
			break
		round += 1
	print('In the end, %d are remian as not matched'%len(final_remain_index))

	# save the mismatched data if any. usually, if no early stopping, all data can be processed and clustered, no remaining data.
	if para['saveFile']:
		file = open(para['output_path']+'/' + 'mismatch.csv', 'w')
		for j in final_remain_index:
			file.write(str(j) + '\t')
			strvec = list(map(str, raw_data[j]))
			file.writelines(' '.join(strvec))
			file.write('\n')

	# get the final cluster index for each sequence data.
	finaclu_result = -1 * np.ones((raw_data.shape[0]))
	label = 0
	for res in overclu_res:
		labelList = res[:, 1]
		cluSize = len(set(labelList))
		for i, va in res:
			if finaclu_result[i] == -1 and va != -1:
				finaclu_result[i] = va + label
		if -1 in labelList:
			label = label + cluSize - 1
		else:
			print("no -1 in labelList, program will stop")
	print("the final cluster number is %d" % len(set(finaclu_result)))
	if len(set(finaclu_result)) != len(all_repres):
		print('error clustering results!!!')
	if -1 in finaclu_result:
		print('remaining -1 in finalcluresult')

	# save all representatives
	np.savetxt(para['rep_path']+'repre_seqs.csv', np.array(all_repres), fmt='%f', delimiter=',')
	print('====================there are ## %d ## clusters=================='%len(all_repres))
	return finaclu_result


def dist_compute(data):
	""" calculate the distance between any two vectors in a matrix.

	Args:
	--------
	data: the data matrix whose distances will be calculated.

	Returns:
	--------
	dist_list: flatten distance list
	"""

	dis = pdist(data, 'euclidean')
	zeroarray = np.zeros(len(dis))
	dist_list = np.maximum(dis, zeroarray) #to avoid negative distance
	return dist_list