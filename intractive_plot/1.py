import pylab as plt


def on_click(event):
    print(event)
    print(event.inaxes == axis)


fig = plt.figure(figsize=(5, 5))
axis = fig.add_subplot(1, 1, 1)
axis.plot(range(10))
fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()
