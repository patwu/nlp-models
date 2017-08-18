#coding:utf-8
import os
import os.path
import re
import linecache

rootdir = "/home/zllin/F/nlp-models-master-wu/nlp-models-master/aclImdb"                                   # 指明被遍历的文件夹

def make_source():
	num = 0
	with open(rootdir + "/contentlist_test1.txt", "w") as f:

		for parent,dirnames,filenames in os.walk(rootdir + "/test/pos"):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字

			for filename in filenames:                        #输出文件信息

				file_dir = os.path.join(parent,filename) #输出文件路径信息
				comment = open(file_dir, 'r')
				commentR = comment.read()
				com = re.sub('\t', '&', commentR)
				#commentR250 = commentR[0:240]
				f.write(str(num) + '\t' + "1" + '\t' + com +"\n")
				num += 1

		print(num)
		for parent,dirnames,filenames in os.walk(rootdir + "/test/neg"):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字

			for filename in filenames:                        #输出文件信息

				file_dir = os.path.join(parent,filename) #输出文件路径信息
				comment = open(file_dir, 'r')
				commentR = comment.read()
				com = re.sub('\t', '&', commentR)
				#commentR250 = commentR[0:240]
				f.write(str(num) + '\t' + "0" + '\t' + com +"\n")
				num += 1

def findtab():
	with open(rootdir + "/contentlist_test1.txt", 'r' + 'w') as f:
		for line in f.readlines():
			num = line.split('\t')[0]
			sentence = line.split('\t')[2]
			try:
				other = line.split('\t')[3]
				print('other' + str(num))

			except:
				continue
			sent = sentence.strip()
			n = re.sub('\t', '(tab)', str(line.split('\t')[2:]))

def mix():
	num = 0
	n_line = 0
	n = 0

	read = open(rootdir + "/contentlist_test1.txt", "r")

	with open(rootdir + "/contentlist_test_mix1.txt", "w") as f:

		for i in read.readlines():
			n_line += 1
		print(n_line)

		for j in range(n_line/2):
			f.write(str(n) + '\t' + linecache.getline(rootdir + "/contentlist_test1.txt", n_line - j - 1).split('\t', 1)[1])
			n += 1
			f.write(str(n) + '\t' + linecache.getline(rootdir + "/contentlist_test1.txt", j+1).split('\t', 1)[1])
			n += 1


make_source()
findtab()
mix()

