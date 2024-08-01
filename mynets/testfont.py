from matplotlib import pyplot as plt
import matplotlib

a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

for i in a:
    print(i)
