#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import pandas as pd
import os
import numpy as np

#***********************************CODE USAGE GUIDE***************************************
#                             		Work for FSE 2018
# Not directly used, should be invoked by cascading_clustering.py                                  
#
# save_results.py is a script that save the results into individual files for manual checking. 
# It mainly save the clustering results of each iteration into files.
# Note that the saving process may be blocked by I/O and increases the time usage.
# Therefore, we use a flag "saveFile" to control whether to save files.
#******************************************************************************************


def saveMatching(para, raw_data, clu_array, curfileIndex, raw_index):
	""" save the matched clusters. work only if saveFile is true

	Args:
	--------
	para: the dictionary of parameters, set in run.py
	raw_data: unweighted raw data. it is used for saving into files, raw data are saved without weighting.
	clu_array: the cluster index list for current data
	curfileIndex:  curfileIndex, flag used for saving
	raw_index: store the sequence index in the raw data, used when saving cluster into files, obtained in loading_all_data()

	Returns:
	--------
	curfileIndex: updated curfileIndex
	"""

	cluResult = list(set(clu_array))
	matcluNum = len(cluResult) - 1
	if -1 not in cluResult:
		matcluNum = matcluNum + 1
	print('------%d clusters are matched (0 to cluster %d) and one more cluster is for the mismatched data'%(matcluNum,matcluNum-1))
	matCluIndeList = [[] for _ in range(matcluNum)]
	# save all the matched sequences, except the mismatched file
	for i, ind in enumerate(clu_array):
		ind = int(ind); i = int(i)
		if ind != -1:
			matCluIndeList[ind].append(raw_index[i])

	# save with multiprocessing, invoke saveSingleFile as one process
	fileIndList = range(curfileIndex, curfileIndex + matcluNum)
	pool = multiprocessing.Pool(para['proc_num'], initializer=init_save_matching, initargs=(raw_data,para, ))
	pool.starmap_async(saveSingleFile, zip(matCluIndeList, fileIndList))  #_async  , chunksize = 4
	pool.close()
	pool.join()
	curfileIndex = curfileIndex + matcluNum
	return curfileIndex


def saveSingleFile(clu, fileindex):
	""" save a clusters of sequence data, used in multiprocess part of saveMatching

	Args:
	--------
	clu: index list of sequence vectors that belong this cluster
	fileindex: used to output the filename as the cluster index
	"""

	datamat = []
	for j in clu:
		row = []
		row.append(j)
		row.extend(raw_data[j,:])
		datamat.append(row)
	pd.DataFrame(np.array(datamat)).to_csv(para['output_path']+'/' + str(fileindex) + '.csv', header=None, index=False)


def init_save_matching(rawData, paras):
	""" initialize some global variables for sharing in multiprocess, used in multiprocess part of saveMatching

	Args:
	--------
	rawData: all raw sequence data, not weighted.
	paras: the dictionary of parameters, set in run.py
	"""
	global raw_data, para
	raw_data = rawData
	para =paras


def deleteAllFiles(dirPath):
	""" delete all files under this dirPath

	Args:
	--------
	dirPath: the folder path whose files would all be deleted
	"""
	fileList = os.listdir(dirPath)
	for fileName in fileList:
		os.remove(dirPath+"/"+fileName)