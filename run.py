#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cascading_clustering as cluster
import time

#***********************************CODE USAGE GUIDE***************************************
#                             		Work for FSE 2018
#                              
# 1. How to run the code?
#    Open a terminal, run the cascading clustering with "python run.py".
#    Make sure that you have Python 3 and all required packages installed.
#
# 2. How to set the parameters?
#    Replace the parameters in the following "para" according to your data
# 
# Notes: multiprocessing is only used to read input files and save output files.
#******************************************************************************************

if __name__ == '__main__':
	para = {
		'seq_folder':  'seq_folder/',    # folder of log sequence matrix file
		'kpi_path':    'kpi_path',       # the path of KPI file
		'proc_num':    16,               # number of processes when loading files and saving files
		'sample_rate': 100,              # sample rate for sampling, 100 represents 1% sample rate
		'thre':        0.3,              # threshold for clustering, and also used when matching the nearest sequence
		'saveFile':    False,            # FLAG to decide whether saving output clusters, it costs a lot if turned on, default False
		'output_path': 'output_path',    # folder for saving output clusters of data
		'rep_path':    'rep_path/',      # path used for savinng all representatives (patterns).
	}
	kpipath = para['kpi_path']

	t1 = time.time()
	rawData, rawIndex, eveOccuMat = cluster.loading_all_data(para)
	kpiList = cluster.load_kpi(kpipath)
	corWeightList = cluster.get_corr_weight(eveOccuMat, kpiList)
	weight_data, weightList = cluster.weighting(rawData, corWeightList)
	finacluResult = cluster.cascading(para, rawData, rawIndex, weight_data)
	t2 = time.time() - t1
	print('the entire time usage is ', t2)