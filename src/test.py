import matplotlib.pyplot as plt
from matplotlib import rc

# LaTeX aktivieren und XCharter einstellen
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['XCharter']})

# Beispielplot
plt.plot([0, 1, 2], [0, 1, 4])
plt.title(r"Beispielplot mit $\texttt{XCharter}$")
plt.xlabel("X-Achse")
plt.ylabel("Y-Achse")
plt.show()
