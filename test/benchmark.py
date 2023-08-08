import os

def run(method, n_values, k, n_leaf, approach, n_test):
	if (method == 2):
		n_leaf = 1
	for n in n_values:
		cmd = f"./test_kdtree {method} {n} {k} {n_leaf} {approach} {n_test}"
		os.system(cmd)

n = [10000, 100000, 1000000, 10000000, 100000000]
k = 100
nl = 12
nt = 10
approach = 0
for method in [3, 0, 1, 2]:
	run(method, n, k, nl, approach, nt)
