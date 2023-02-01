###CONDITIONAL HISTOGRAM
import matplotlib.pyplot as plt
def hist_error(error, bin):
    h = error.max()/bin
    hist = np.asarray([[1e-6,0],
                  [1e-3,0],
                   [1.0,0]])
    for idx in range(1,bin+2):
        val = int(h*idx)
        hist = np.append(hist, [[val,0]], axis = 0)
    for x in range(len(error)):
        for y in range(len(hist)):
            if y==0 and error[x]<hist[0,0]:
                hist[0,1] = hist[0,1]+1
            else:
                if hist[y-1,0]<error[x]<hist[y,0]:
                    hist[y,1] = hist[y,1]+1
    return hist

hist = hist_error(error,10)
plt.bar(hist[:,0],hist[:,1],5)

#3D PLOT
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(source_array_transformed[:1000,0],source_array_transformed[:1000,1],source_array_transformed[:1000,2])
plt.show()