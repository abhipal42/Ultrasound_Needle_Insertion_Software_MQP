import matplotlib.pyplot as plt

total=1500
y = []
x = []
Per=.23


for i in range(1,11):
    Per = (total-250)*Per/total
    y.append(Per)
    x.append(i)

plt.plot(x, y)
plt.show()