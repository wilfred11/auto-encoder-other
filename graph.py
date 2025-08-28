import ffmpeg
import pandas as pd
from celluloid import Camera
from matplotlib import pyplot as plt, animation


def build_embedding_animation(encoded_data_per_eval, iterations_per_eval=100, model_name=""):
    fig, ax = plt.subplots()

    camera = Camera(fig)

    for idx, encoding in enumerate(encoded_data_per_eval):

        encoding = pd.DataFrame(encoding, columns=["x", "y", "class"])
        encoding = encoding.sort_values(by="class")
        encoding["class"] = encoding["class"].astype(int).astype(str)

        for grouper, group in encoding.groupby("class"):
            plt.scatter(x=group["x"], y=group["y"], label=grouper, alpha=0.8, s=5)

        ax.text(0.4, 1.01, f"Step {idx * iterations_per_eval}", transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        camera.snap()

    plt.close()
    #interval = 200, repeat = True,
    #repeat_delay = 500
    anim = camera.animate(blit=True, interval=10)

    plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
    FFwriter = animation.FFMpegWriter(fps=6, extra_args=['-vcodec', 'libx264'])
    anim.save('graph/'+model_name+'/animation.mp4', writer=FFwriter)

def build_embedding_plot(encoding, title,model_name):
    encoding = pd.DataFrame(encoding, columns=["x", "y", "class"])
    encoding = encoding.sort_values(by="class")
    encoding["class"] = encoding["class"].astype(int).astype(str)

    leg=[]
    for grouper, group in encoding.groupby("class"):
        plt.scatter(x=group["x"], y=group["y"], label=grouper, alpha=0.8, s=5)
        leg.append(grouper)

    plt.legend(leg)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.show()
    plt.savefig("graph/"+model_name+"/graph_.png")

