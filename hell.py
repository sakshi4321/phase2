import pickle

with open("1213.dat","rb") as f:
	obj=pickle.load(f)
	print(obj)
