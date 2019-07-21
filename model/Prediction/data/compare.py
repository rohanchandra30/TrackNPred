import numpy as np
import scipy.io as scp



# D = (scp.loadmat(mat_file)['traj']).astype(float)
matD = (scp.loadmat("TestSetTRAF.mat")['traj'])
matT = (scp.loadmat("TestSetTRAF.mat")['tracks'])

train = np.load("./TRAF/TRAFTrainSet.npy", allow_pickle=True)
val = np.load("./TRAF/TRAFValSet.npy", allow_pickle=True)
test = np.load("./TRAF/TRAFTestSet.npy", allow_pickle=True)

# print(np.shape(matD))
# print(np.shape(matT))
# print(train[1].keys())
# ids = []
# for l in matD:
# 	ids.append(l[1])
# print(np.unique(ids))

# for (i, l) in enumerate(matT[1]):
	# if l.size != 0:
		# print(i)







# print(np.shape(matT[1][93]))
# print(np.shape(matT[1][92]))

traj = train[0]
tracks = train[1]
maxdid = max(tracks.keys())

maxvehicleset = []
for v in tracks.values():
	maxvehicleset.append(max(v.keys()))
maxvehicle = int(max(maxvehicleset))



print(maxvehicle)
print(range(maxvehicle))

temp = {}

for k in tracks.keys():
	# print(k)
	temp[k] = np.full(maxvehicle, .0, dtype=object)

	for i in range(maxvehicle):
		if i in tracks[k].keys():
			temp[k][i] = tracks[k][i]
		else:
			temp[k][i] = np.empty((1,0))


# print(temp[12][33])
# print(matT[1][93])

print(np.shape(temp))
print(np.shape(matT))




# print(np.shape(matT))
# print(np.shape(train[1][12]))
# print(train[1][12][33])
# print(train[1][12].keys())

# ids = []
# for l in train[0]:
# 	ids.append(l[1])
# print(np.unique(ids))






# print(matT[0][0].size)


# temp = np.array([[1,3],[2,4]])

# for k, v in data[1].items():
# 	print(k)
# 	temp = np.append(temp, [[1,3]], axis=0)

# print(temp)
# print(type(matT))
# print(type(data[1]))