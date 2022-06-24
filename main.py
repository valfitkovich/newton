import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

alpha = 0.0
eps = 1e-8

# --правки внесены
def f(arg):
    return np.array([arg[1],
                     np.sign(arg[3] - (1/(1+alpha*(arg[0])*(arg[0])*(arg[1])*(arg[1])))),
                     (-2*alpha* (np.sign(arg[3] - (1/(1+alpha*(arg[0])*(arg[0])*(arg[1])*(arg[1]))))) * arg[0]*arg[1]*arg[1])/((1+alpha*(arg[0])*(arg[0])*(arg[1])*(arg[1]))*(1+alpha*(arg[0])*(arg[0])*(arg[1])*(arg[1]))),
                     (-2*alpha* (np.sign(arg[3] - (1/(1+alpha*(arg[0])*(arg[0])*(arg[1])*(arg[1]))))) * arg[0]*arg[0]*arg[1])/((1+alpha*(arg[0])*(arg[0])*(arg[1])*(arg[1]))*(1+alpha*(arg[0])*(arg[0])*(arg[1])*(arg[1]))) - arg[2]])


# функция, которая врзвращает значения в точке t=1 функций x1 и p2 по заданным значениям в точке t=0
# --правки для моей задачи внесены
def f_at_1(arg):
    val = np.array([0, 0, arg[0], arg[1]])
    #print(val)
    step = 0.001
    t = 0
    count = 0
    while t - 1 < eps:
        k1 = step * f(val)
        k2 = step * f(val + 0.25 * k1)
        k3 = step * f(val + 0.5 * k1)
        k4 = step * f(val + k1 / 7 + 2 * k2 / 7 + k3 / 14)
        k5 = step * f(val + 3*k1 / 8 - k3 / 2 + 7 * k4 / 8)
        k6 = step * f(val - 4 * k1 / 7 + 12 * k2 / 7 - 2 * k3 / 7 - k4 + 8 * k5 / 7)
        #k1 = step*f(val)
        #k2 = step*f(val+0.5*k1)
        #k3 = step*f(val+0.5*k2)
        #k4 = step*f(val+k3)
        #if (val[3]- 1/(1 + alpha*val[0]*val[0]*val[1]*val[1]))*(val[3]- 1/(1 + alpha*val[0]*val[0]*val[1]*val[1])+(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6)<0:
        if (val[3] - 1 / (1 + alpha * val[0] * val[0] * val[1] * val[1])) * (
                val[3] - 1 / (1 + alpha * val[0] * val[0] * val[1] * val[1]) + (k1[3] + 7 * (k1[3] + k6[3]) / 90 + 16 * (k2[3] + k5[3]) / 45 - k3[3] / 3 + 7 * k4[3] / 15)) < 0:
            step /= 10
            count += 1
            if count > 10:
                #val += (k1 + 2 * k2 + 2 * k3 + k4) / 6
                val += 7 * (k1 + k6) / 90 + 16 * (k2 + k5) / 45 - k3 / 3 + 7 * k4 / 15
                t += step
                count = 0
                step = 0.001
        else:
            #val += (k1+2*k2+2*k3+k4)/6
            val += 7 * (k1 + k6) / 90 + 16 * (k2 + k5) / 45 - k3 / 3 + 7 * k4 / 15
            t += step
    val[0] += 0.4583333333333333
    return np.array([val[0], val[3]])


# функция, возвращающая решение системы уравнений по модифицированному методу Ньютона
# не понимаю, куда всунуть 11/24
def NewtonSolve(cur_a0b0):
    dx = np.array([0.01, 0])
    dy = np.array([0, 0.01])
    jacob = np.array([f_at_1(cur_a0b0 + dx) / np.linalg.norm(dx), f_at_1(cur_a0b0 + dy) / np.linalg.norm(dy)]).transpose()
    jacob = np.linalg.inv(jacob)
    x = cur_a0b0 - np.dot(jacob, f_at_1(cur_a0b0))
    i = 0
    #print(np.linalg.norm(x - init_ab) - eps > 0)
    #print(jacob)
    while np.linalg.norm(x - cur_a0b0) > eps:
        cur_a0b0 = x
        x = cur_a0b0 - np.dot(jacob, f_at_1(cur_a0b0))
        i += 1
        print(np.linalg.norm(x-cur_a0b0))
    return x


# функция, решающая систему дифференциальных уравнений. В ней реализована модификация шага метода Рунге-Кутты при приближении к точке смены знака р2
# --правки внесены, кроме "при приближении к смене знака р2"
# --правки внесены
def ODESolve(arg):
    t = []
    val = []
    t_cur = 0
    step = 0.0001
    val_cur = np.array([0, 0, arg[0], arg[1]])
    t.append(t_cur)
    val_cur_copy = np.array([val_cur[0], val_cur[1], val_cur[2], val_cur[3]])
    val.append(val_cur_copy)
    count = 0
    while t_cur - 1 < eps:
        #k1 = step*f(val_cur)
        #k2 = step*f(val_cur+0.5*k1)
        #k3 = step*f(val_cur+0.5*k2)
        #k4 = step*f(val_cur+k3)
        k1 = step * f(val_cur)
        k2 = step * f(val_cur + 0.25 * k1)
        k3 = step * f(val_cur + 0.5 * k1)
        k4 = step * f(val_cur + k1 / 7 + 2 * k2 / 7 + k3 / 14)
        k5 = step * f(val_cur + 3 * k1 / 8 - k3 / 2 + 7 * k4 / 8)
        k6 = step * f(val_cur - 4 * k1 / 7 + 12 * k2 / 7 - 2 * k3 / 7 - k4 + 8 * k5 / 7)
        if (val_cur[3] - 1/(1 + alpha*val_cur[0]*val_cur[0]*val_cur[1]*val_cur[1]))*(val_cur[3] - 1/(1 + alpha*val_cur[0]*val_cur[0]*val_cur[1]*val_cur[1])+(7 * (k1[3] + k6[3]) / 90 + 16 * (k2[3] + k5[3]) / 45 - k3[3] / 3 + 7 * k4[3] / 15)) < 0:
            step *= 0.1

            val_cur += 7 * (k1 + k6) / 90 + 16 * (k2 + k5) / 45 - k3 / 3 + 7 * k4 / 15
            t_cur += step
            val_cur_copy = np.array([val_cur[0], val_cur[1], val_cur[2], val_cur[3]])
            val.append(val_cur_copy)
            t.append(t_cur)

            # count += 1
            # if count > 10:
            #     #val_cur += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            #     val_cur += 7 * (k1 + k6) / 90 + 16 * (k2 + k5) / 45 - k3 / 3 + 7 * k4 / 15
            #     t_cur += step
            #     val_cur_copy = np.array([val_cur[0], val_cur[1], val_cur[2], val_cur[3]])
            #     val.append(val_cur_copy)
            #     t.append(t_cur)
            #     count = 0
            #     step = 0.0001
        else:
            #val_cur += (k1+2*k2+2*k3+k4)/6
            val_cur += 7 * (k1 + k6) / 90 + 16 * (k2 + k5) / 45 - k3 / 3 + 7 * k4 / 15
            t_cur += step
            val_cur_copy = np.array([val_cur[0], val_cur[1], val_cur[2], val_cur[3]])
            val.append(val_cur_copy)
            t.append(t_cur)
    val = pd.DataFrame(val, index=t, columns=["x1", "x2", "p1", "p2"])
    return val

#В этом цикле двигаемся по параметру в случае, когда он не превосходит 10. Если же это не так, ниже строка с вызовом функции от угаданнного начального приближения при \alpha=15
alpha = 0.0
a0b0 = np.array([1.0215078369104983, 1.0215078369104983])
while alpha < 11.0:
    a0b0 = NewtonSolve(a0b0)
    alpha += 0.01
print(a0b0)
sol = ODESolve(a0b0)
t = sol.index

#графики
fig, ax = plt.subplots()
ax.plot(t, sol["x1"], t, sol["x2"], t, sol["p1"], t, sol["p2"])
ax.grid()
plt.show()



#запись в файл Excel
sol.to_excel(r'C:\Users\Лерочка\PycharmProjects\evm3\sol11.xlsx')