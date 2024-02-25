import matplotlib.pyplot as plt
import numpy as np

def my_linfit(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(x_i**2 for x_i in x)
    sum_xy = sum(x[i] * y[i] for i in range(len(x)))
    b = (sum_y * sum_x_squared - sum_xy * sum_x) / (sum_x_squared * len(x) - sum_x**2)
    a = (-b * sum_x + sum_xy) / sum_x_squared
    return a, b

def main():
    x = []
    y = []

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 5)  
    ax.set_ylim(0, 3)
    points, = ax.plot([], [], 'kx')
    line_fit, = ax.plot([], [], 'r-')

    def onclick(event):
        if event.button == 1:  # Left-click
            x.append(event.xdata)
            y.append(event.ydata)
            points.set_data(x, y)
            plt.draw()

        elif event.button == 3:  # Right-click
            a, b = my_linfit(x, y)
            xp = np.arange(-2, 5, 0.1)
            line_fit.set_data(xp, a * xp + b)
            plt.draw()
            print(f"My fit: a = {a} and b = {b}")
            fig.canvas.mpl_disconnect(cid)  # Disconnect the event handler to stop collecting data

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    main()
