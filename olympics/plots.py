"""
# keep synch values for arms and legs o
# remove missing values
np.savetxt('synch.csv', synchrony_totalvideo, delimiter=',')
np.savetxt('heights.csv', heights_totalvideo, delimiter=',')
np.savetxt('distances.csv', distances_totalvideo, delimiter=',')

print("synchrony_totalvideo\n", synchrony_totalvideo)
print("heights_totalvideo\n", heights_totalvideo)
print("distances_totalvideo\n", distances_totalvideo)

for i in range(1, synchrony_totalvideo.shape[1]):
    plt.plot([row[i] for row in synchrony_totalvideo])
plt.show()

for i in range(1, heights_totalvideo.shape[1]):
    plt.plot([row[i] for row in heights_totalvideo])
plt.show()

for i in range(1, distances_totalvideo.shape[1]):
    plt.plot([row[i] for row in distances_totalvideo])
plt.show()

for i in range(1, synchrony_totalvideo.shape[1]):
    plt.plot([row[i] for row in synchrony_totalvideo])

# replace missing values with mean values
synchrony_totalvideo_optimized = data_optimization.fill_missing_values_synchrony(synchrony_totalvideo)

# plot synch values with frames as x axis and vals on y
plt.plot(synchrony_totalvideo_optimized.T[0],
         synchrony_totalvideo_optimized.T[1:].T)  # plotting w/o loop needs y as column
plt.legend(tuple((pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][0]] + " to " +
                  pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][1]]) for t in
                 range(len(pose_estimation.POSE_PAIRS))))
plt.show()

# version 2
for i in range(1, synchrony_totalvideo_optimized.T.shape[0]):
    plt.plot(synchrony_totalvideo_optimized.T[0],
             synchrony_totalvideo_optimized.T[i])  # plotting in loop needs y as row
plt.legend(tuple((pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][0]] + " to " +
                  pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][1]]) for t in
                 range(len(pose_estimation.POSE_PAIRS))))
plt.show()

# plot synch values for relevant body parts only
plt.figure()
plt.plot(synchrony_totalvideo_optimized.T[0],
         np.array(list(synchrony_totalvideo_optimized.T[x + 1] for x in (2, 3, 4, 5, 7, 8, 10, 11))).T)
plt.legend(tuple(((pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][0]] +
                   " to " + pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][1]])
                  for t in (2, 3, 4, 5, 7, 8, 10, 11))))


# plot specific body part synchrony
plt.plot(synchrony_totalvideo_optimized.T[0], synchrony_totalvideo_optimized.T[1])
plt.legend(tuple([pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[0][0]]+
                  " to "+pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[0][1]]]))
plt.show()

"""