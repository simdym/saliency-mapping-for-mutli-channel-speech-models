import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
from .transcript import Transcript

def plot_transcript(
        axes: matplotlib.axes.Axes,
        transcript: Transcript,
        start: float,
        end: float,
        fontsize: float=5,
        text_rotation: float=90
    ):
    """
    Plots a transcript on the given axes.
    """
    ylim = axes.get_ylim()

    for element in transcript.get_elements_between(start, end, include_edge=True):
        x_pos = element.start-start
        if x_pos < 0:
            x_pos = 0
        elif x_pos > (end-start):
            x_pos = (end-start)

        y_range = ylim[1] - ylim[0]
        axes.text(x_pos, ylim[0] + 0.03 * y_range, element.content, fontsize=fontsize, rotation=text_rotation)
        axes.axvline(x_pos, color="black", linewidth=0.5)
        axes.axvline(x_pos, color="black", linewidth=0.5)


def label_x_with_transcript(
        axes: matplotlib.axes.Axes,
        transcript: Transcript,
        start: float,
        end: float,
        fontsize: int=5,
        pos: str="left",
        rel_fs: float=1.0, # frequency / audo sample rate
        upper: bool=False,
        rotation: float=90.0,

    ):
    """
    Labels the x-axis of the given axes with the transcript.
    """
    ticks = []
    labels = []
    for element in transcript.get_elements_between(start, end, include_edge=True):
        if pos == "left":
            tick = (element.start-start) / rel_fs
        elif pos == "right":
            tick = (element.end-start) / rel_fs
        elif pos == "center":
            tick = ((element.start+element.end) / 2 - start) / rel_fs
        else:
            raise ValueError("pos must be left, right or center")

        if tick < 0:
            tick = 0
        elif tick > (end-start) / rel_fs:
            tick = (end-start) / rel_fs

        ticks.append(tick)
        labels.append(element.content)

    if not upper:
        axes.set_xticks(ticks, labels, rotation=rotation, ha=pos, fontsize=fontsize)
    else:
        axes_upper = axes.twiny()
        axes_upper.set_xlim(axes.get_xlim())
        axes_upper.set_xticks(ticks, labels, rotation=rotation, ha=pos, fontsize=fontsize)


if __name__ == "__main__":
    transcript = Transcript()
    transcript.add_element(0, 1, "Hello")
    transcript.add_element(1, 2, "World")
    transcript.add_element(2, 3, "This")
    transcript.add_element(3, 4, "is")
    transcript.add_element(4, 5, "a")
    transcript.add_element(5, 6, "test")
    transcript.add_element(6, 7, "transcript")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot([1,2,3,4,5,6,7])
    plot_transcript(ax2, transcript, 1, 6)
    label_x_with_transcript(ax2, transcript, 1, 6, pos="center")
    fig.savefig("transcript_utils//test.png")