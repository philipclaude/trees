from matplotlib import pyplot as plt
import json

plt.rcParams["font.family"] = "Futura"
plt.rcParams["font.size"] = 12
ext = 'png'
resolution = 144
dst = "figures"

method_name = {0: "balanced", 1: "left-balanced-leaf-full",
               2: "left-balanced-leaf-empty", 3: "nanoflann"}

k = 10
power = [5, 6, 7]
sizes = [10 ** p for p in power]
build = {}
query = {}
memory = {}
for m in [0, 1, 2, 3]:
    nl = 12
    if m == 2:
        nl = 1
    t_b = []
    t_q = []
    mem = []
    for n in power:
        filename = f"results-m{m}-n{n}-k{k}-l{nl}.json"
        print(f"analyzing: {filename}")
        with open("results/" + filename) as f:
            data = json.loads(f.read())
            t_b.append(data["t_build"])
            t_q.append(data["t_knn"])
            mem.append(data["memory"])

    method = method_name[m]
    build[method] = t_b
    query[method] = t_q
    memory[method] = mem

print(build)
print(query)
print(memory)

total = {}
for method in build:
    assert method in query
    total[method] = [build[method][i] + query[method][i]
                     for i in range(len(build[method]))]

plt.figure()
h_b = plt.loglog(sizes, build["balanced"], 'o-')
h_n = plt.loglog(sizes, build["nanoflann"], 'o-')
h_lf = plt.loglog(sizes, build["left-balanced-leaf-full"], 'o-')
h_le = plt.loglog(sizes, build["left-balanced-leaf-empty"], 'o-')
plt.xlabel('# points', fontsize=12)
plt.title('build time (s)', fontsize=12)
plt.legend(handles=[h_b[0], h_lf[0], h_le[0], h_n[0]], labels=[
           'balanced', 'left-bal. (12 pts/leaf)', 'left-bal. (1 pt/leaf)' 'nanoflann'])
# plt.annotate('nanoflann', weight='bold', xy=(
#     10 ** 5, 0.04), color=h_n[0].get_color())
# plt.annotate('balanced', weight='bold',
#              xy=(10 ** 7, 0.2), color=h_b[0].get_color())
# plt.annotate('left-balanced\n(max 12 points / leaf)', weight='bold', xy=(10 ** 7.5, 2.5), xytext=(10 ** 6, 4),
#              arrowprops=dict(shrink=0.01, width=2, edgecolor='black', facecolor=h_lf[0].get_color(), headwidth=8), color=h_lf[0].get_color())
# plt.annotate('left-balanced\n(1 point / leaf)', weight='bold', xy=(10 ** 6.025, 0.03), xytext=(10 ** 6.25, 0.004),
#              arrowprops=dict(shrink=0.01, width=2, edgecolor='black', facecolor=h_le[0].get_color(), headwidth=8), color=h_le[0].get_color())
plt.savefig(f"{dst}/build-k{k}.{ext}", dpi=resolution)

plt.figure()
h_b = plt.loglog(sizes, query["balanced"], 's-')
h_n = plt.loglog(sizes, query["nanoflann"], 'o--')
h_l = plt.loglog(sizes, query["left-balanced-leaf-full"], '^-')
plt.xlabel('# points', fontsize=12)
plt.title(f"k = {k}, query time (s)", fontsize=12)
plt.legend(handles=[h_b[0], h_l[0], h_n[0]], labels=[
           'balanced', 'left-balanced', 'nanoflann'])
plt.savefig(f"{dst}/query-k{k}.{ext}", dpi=resolution)

plt.figure()
h_b = plt.loglog(sizes, total["balanced"], 's-')
h_n = plt.loglog(sizes, total["nanoflann"], 'o--')
h_l = plt.loglog(sizes, total["left-balanced-leaf-full"], '^-')
plt.xlabel('# points', fontsize=12)
plt.title(f"k = {k}, build + query time (s)", fontsize=12)
plt.legend(handles=[h_b[0], h_l[0], h_n[0]], labels=[
           'balanced', 'left-balanced', 'nanoflann'])
plt.savefig(f"{dst}/total-k{k}.{ext}", dpi=resolution)

# plt.show()
