import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
x = [i for i in np.arange(0.0, 1.1, 0.1)]
print(x)
# y_1 = [50, 60, 70]
# y_2 = [20, 30, 40]
y_1 = list(map(float, '51939	34705	29522	27990	30965	27068	29104 31289	28183	27637	32103'.split()))
plt.plot(x, y_1, marker='x')
# plt.plot(x, y_2, marker='^')
# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
# plt.rcParams["axes.unicode_minus"]=False
plt.rcParams['figure.dpi'] = 300


x_major_locator=MultipleLocator(0.1)
#把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(10)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)




plt.xlim([-0.05, 1.05])
# plt.ylim([0, max(y_1+y_2) + 10])
plt.xlabel('Balance current and future $\epsilon$')
plt.ylabel('Cost')
plt.title('The impact of $\epsilon$')
# plt.legend(['sample 1',], loc='upper left')
plt.gcf().set_dpi(300)
plt.show()
# plt.savefig('./epsilon.pdf')