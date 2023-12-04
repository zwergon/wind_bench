
import matplotlib.pyplot as plt

class Prediction:

    def __init__(self, file, predicted, actual, title=None, x_label=None, y_label=None) -> None:
        self.file = file
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.predicted = predicted
        self.actual = actual

    def plot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.predicted, self.actual)
        if self.y_label is not None:
            ax.set_ylabel(self.y_label)
        if self.x_label is not None:
            ax.set_xlabel(self.x_label)
        return fig
        

if __name__ == "__main__":

    import numpy as np
    X = np.arange(-4., 4., 0.1)
    s_x = np.sin(X)
    c_x = np.cos(X)
    prediction = Prediction("circle.png", s_x, c_x, y_label="s_x")
    fig = prediction.plot()
    plt.show()