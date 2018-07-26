#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import pandas as pd
import os
import numpy as np
import time
import glob
import multiprocessing

#***********************************CODE USAGE GUIDE***************************************
#                             		Work for FSE 2018
# Not directly used, should be invoked by cascading_clustering.py    


# data_loader.py is a script that loads the datasets using multiple processes.
# Description: 
# It loads the log sequence matrix files and a KPI file into the memory. Furthermore, 
# duplicate events are removed after loading. To facilitate the loading process, we utilize
# the Python multiprocessing to load a large number of log sequence matrix files. 
#******************************************************************************************



def loading_all_data(para):
	""" load all log sequence matrixs, remove duplicates, and count the number of
	log sequences that contain an event (used for correlation weighting in section 3.2)

	Args:
	--------
	para: the dictionary of parameters, set in run.py

	Returns:
	--------
	allrawData: loaded all log sequence matrix, these matrix are merged into one big matrix of (N, M).
				N is the number of all log sequences, M is event number.
	rawIndex:   index list that used to mark which log sequences are clustered.
	eveOccuMat: count the number of log sequences that contain each event, it will be used for weighting
	"""

	t0 = time.time()
	# find the all log sequence matrix files.
	path = para['seq_folder']
	fileList = glob.glob(path + 'timeInter_*.csv')
	fileNumList = []
	for file in fileList:
		fileNum = file.replace(path + 'timeInter_', '').replace('.csv', '')
		fileNumList.append(int(fileNum))
	print("there are %d log sequence files files found"%(len(fileNumList)))
	newfileList = []
	for x in sorted(fileNumList):
		newfileList.append(path+ 'timeInter_'+str(x)+'.csv')

	# load all the files using multiprocessing.
	print('start loading data')
	pool = multiprocessing.Pool(para['proc_num'])
	rawdataList = pool.map(load_single_file, newfileList)
	pool.close()
	pool.join()
	allrawData = np.vstack(rawdataList)

	# index used to mark which log sequences are already processed
	rawIndex = range(0, allrawData.shape[0])

	# count the number of log sequences that contain each event, it will be used for weighting
	eveOccuMat = []
	for inter_data in rawdataList:
		eveOccuMat.append(np.sum(inter_data, axis = 0))
	eveOccuMat = np.array(eveOccuMat)
	print('Step 1. Data Loading, the raw input data size is %d, it takes %f' % (allrawData.shape[0], time.time() - t0))
	return allrawData, rawIndex, eveOccuMat


def load_kpi( kpipath):
	""" load the KPI data

	Args:
	--------
	kpipath:  data path of KPI

	Returns:
	--------
	kpiList:  list of KPIs, one KPI value per time interval.
	"""

	df = pd.read_csv(kpipath, dtype= int, header=None)
	kpiList = df.as_matrix()
	return kpiList


def load_single_file(filepath):
	""" load one log sequence matrix from the file path, and duplicate events are removed.

	Args:
	--------
	filepath:  file path of a log sequences matrix

	Returns:
	--------
	rawData:  log sequences matrix (duplicates removed)
	"""

	df = pd.read_csv(filepath, header=None)
	rawData = df.as_matrix()
	rawData[rawData > 1] = 1
	return rawData
