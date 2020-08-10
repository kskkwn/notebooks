import pylab as plt


def button_press(event):
    print(event)


def button_release(event):
    print(event)


fig = plt.figure(figsize=(5, 5))
axis = fig.add_subplot(1, 1, 1)
axis.plot(range(10))
fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()
