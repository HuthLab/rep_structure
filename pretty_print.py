import joblib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from matplotlib.lines import Line2D


class AxesDecorator():
    def __init__(self, ax, size="5%", pad=0.05, ticks=[1,2,3], spacing=0.05,
                 color="k"):
        self.divider= make_axes_locatable(ax)
        self.ax = self.divider.new_vertical(size=size, pad=pad, sharex=ax, pack_start=True)
        ax.figure.add_axes(self.ax)
        self.ticks=np.array(ticks)
        self.d = np.mean(np.diff(ticks))
        self.spacing = spacing
        self.get_curve()
        self.color=color
        for x0 in ticks:
            self.plot_curve(x0)
        self.ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)
        self.ax.tick_params(axis='x', which=u'both',length=0)
        ax.tick_params(axis='x', which=u'both',length=0)
        for direction in ["left", "right", "bottom", "top"]:
            self.ax.spines[direction].set_visible(False)
        self.ax.set_xlabel(ax.get_xlabel(), fontsize= 24)
        ax.set_xlabel("")
        self.ax.set_xticks(self.ticks)

    def plot_curve(self, x0):
        x = np.linspace(x0-self.d/1.*(1-self.spacing),x0+self.d/2.*(2-self.spacing), 50 )
        self.ax.plot(x, self.curve, c=self.color)

    def get_curve(self):
        lx = np.linspace(-np.pi/2.+0.05, np.pi/2.-0.05, 25)
        tan = np.tan(lx)*10
        self.curve = np.hstack((tan[::-1],tan))
        return self.curve

def bracket_X(ax, offset, pos=[0,0], color="black", scalex=1, scaley=1, text="",textkw = {}, linekw = {}, rotate=90):
    x = np.array([0, 0.05, 0.45,0.5])
    y = np.array([0,-0.01,-0.01,-0.02])
    x = np.concatenate((x,x+0.5))
    y = np.concatenate((y,y[::-1]))
    ax.plot(x*scalex+pos[0], -y*scaley+pos[1], clip_on=False,
            transform=ax.get_xaxis_transform(), **linekw)
    ax.text(pos[0]+0.5*scalex+offset, (y.min()+0.042)*scaley+pos[1], text,
                transform=ax.get_xaxis_transform(), size=48,  rotation=rotate,
                ha="center", va="bottom", color=color,weight="bold", **textkw)

def bracket_Y(ax, offset, color="black", textoffset=0, pos=[0,0], scalex=1, scaley=1, text="",textkw = {}, linekw = {}, rotate=0):
    x = np.array([0, 0.05, 0.45,0.5])
    y = np.array([0,-0.01,-0.01,-0.02])
    x = np.concatenate((x,x+0.5))
    y = np.concatenate((y,y[::-1]))
    ax.plot(y*scaley+pos[1], x*scalex+pos[0], clip_on=False,
            transform=ax.get_xaxis_transform(), **linekw)
    ax.text((y.min()-0.005)*scaley+pos[1]+textoffset, pos[0]+0.5*scalex-offset, text,
                transform=ax.get_xaxis_transform(), size=48, rotation=rotate,
                ha="center", va="bottom", color=color,weight="bold", **textkw)


def pretty_print_mat(affinity):
    scale = [6] * 8 + [2] * 12 + [1] * 24 + [1] * 18 + [2] * 13 + [2] * 13 + [3] * 6  + [3] * 6
    arr  = []
    for ei, i in enumerate(scale):
        for k in range(i):
            lst = []
            for ej, j in enumerate(scale):
                for l in range(j):
                    lst.append(affinity[ei][ej])
            arr.append(lst)
    arr = np.array(arr)
    arr_len = arr.shape[0]
    fig = plt.figure(figsize=(100,120))
    ax = fig.add_subplot(221, label='k')

    cax = ax.matshow(arr, interpolation='nearest', vmin=0, vmax=0.1)
    cb = fig.colorbar(cax, shrink=0.5, orientation='horizontal', pad=0.004)
    cb.ax.tick_params(labelsize=60)
    plt.yticks([])
    plt.xlim(-0.5,arr_len - 0.5)
    plt.ylim(arr_len - 0.5,-0.5)
    ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False)
    ax.tick_params(axis="y", bottom=False, top=False, labelbottom=False, labeltop=False)



    bracket_X(ax,0.1,text="GloVe",  color='green',  scalex=5, pos=[0,1.005], linekw=dict(color="black", lw=2) )
    bracket_X(ax,0.1, text="BERT-E", color='green', scalex=5, pos=[6,1.005], linekw=dict(color="black", lw=2) )
    bracket_X(ax,0.1, text="FLAIR", color='green',scalex=5, pos=[12,1.005], linekw=dict(color="black", lw=2)  )
    bracket_X(ax,0.1, text="POS",  color='red', scalex=5, pos=[18,1.005], linekw=dict(color="black", lw=2) )
    bracket_X(ax,0.1, text="CHUNK",  color='red',scalex=5, pos=[24,1.005], linekw=dict(color="black", lw=2) )
    bracket_X(ax,0.1, text="NER",color='blue',  scalex=5, pos=[30,1.005], linekw=dict(color="black", lw=2) )
    bracket_X(ax,0.1, text="FRAME",   color='blue', scalex=5, pos=[36,1.005], linekw=dict(color="black", lw=2) )
    bracket_X(ax,0.1, text="NWE", scalex=5, pos=[42,1.005], linekw=dict(color="black", lw=2) )
    bracket_X(ax,0.1, text="GPT-2 S", color='orange', scalex=23, pos=[48,1.005], linekw=dict(color="black", lw=2), rotate=0 )
    bracket_X(ax,0.1, text="GPT-2 M", color='coral',scalex=23, pos=[72,1.005], linekw=dict(color="black", lw=2), rotate=0  )
    bracket_X(ax,0.1, text="Trans-XL", color='fuchsia', scalex=17, pos=[96,1.005], linekw=dict(color="black", lw=2), rotate=0  )
    bracket_X(ax,0.1, text="BERT",  color='cadetblue', scalex=25, pos=[114,1.005], linekw=dict(color="black", lw=2), rotate=0  )
    bracket_X(ax,0.1, text="ALBERT", color='grey', scalex=25, pos=[140,1.005], linekw=dict(color="black", lw=2), rotate=0  )
    bracket_X(ax,-1, text="Eng$\Rightarrow$Zh", color='mediumvioletred', scalex=17, pos=[166,1.005], linekw=dict(color="black", lw=2), rotate=0  )
    bracket_X(ax,0.1, text="Eng$\Rightarrow$De", color='olive', scalex=17, pos=[184,1.005], linekw=dict(color="black", lw=2), rotate=0  )


    bracket_Y(ax, 0.047, text="Eng$\Rightarrow$De",color='olive', scaley=256, scalex=17/sum(scale), pos=[0.5/sum(scale),-1], linekw=dict(color="black", lw=2), rotate=90)
    bracket_Y(ax, 0.041, text="Eng$\Rightarrow$Zh",color='mediumvioletred', scaley=256, scalex=17/sum(scale), pos=[18.5/sum(scale),-1], linekw=dict(color="black", lw=2), rotate=90)
    bracket_Y(ax, 0.039, text="ALBERT",color='grey', scaley=256, scalex=25/sum(scale), pos=[36.5/sum(scale),-1], linekw=dict(color="black", lw=2), rotate=90)
    bracket_Y(ax, 0.026, text="BERT", color='cadetblue',scaley=256, scalex= 25/sum(scale), pos=[62.5/sum(scale),-1], linekw=dict(color="black", lw=2), rotate=90)
    bracket_Y(ax, 0.047,  text="Trans-XL",color='fuchsia', scaley=256, scalex= 17/sum(scale), pos=[88.5/sum(scale),-1], linekw=dict(color="black", lw=2), rotate=90)
    bracket_Y(ax, 0.044, text="GPT-2 M", color='coral',scaley=256, scalex= 23/sum(scale), pos=[106.5/sum(scale),-1], linekw=dict(color="black", lw=2), rotate=90)
    bracket_Y(ax, 0.04, text="GPT-2 S", color='orange',scaley=256, scalex= 23/sum(scale), pos=[130.5/sum(scale),-1], linekw=dict(color="black", lw=2), rotate=90)
    bracket_Y(ax, 0.01,textoffset=-3.5, text="NWE", scaley=256, scalex= 5/sum(scale), pos=[154.5/sum(scale),-1], linekw=dict(color="black", lw=2))
    bracket_Y(ax, 0.01,textoffset=-6, text="FRAME",   color='blue', scaley=256, scalex= 5/sum(scale), pos=[160.5/sum(scale),-1], linekw=dict(color="black", lw=2))
    bracket_Y(ax, 0.01,textoffset=-3.5, text="NER",color='blue', scaley=256, scalex= 5/sum(scale), pos=[166.5/sum(scale),-1], linekw=dict(color="black", lw=2))
    bracket_Y(ax, 0.01,textoffset=-6,text="CHUNK",  color='red', scaley=256, scalex= 5/sum(scale), pos=[172.5/sum(scale),-1], linekw=dict(color="black", lw=2))
    bracket_Y(ax, 0.01,textoffset=-3.5,text="POS",  color='red', scaley=256, scalex= 5/sum(scale), pos=[178.5/sum(scale),-1], linekw=dict(color="black", lw=2))
    bracket_Y(ax, 0.01,textoffset=-5, text="FLAIR", color='green', scaley=256, scalex= 5/sum(scale), pos=[184.5/sum(scale),-1], linekw=dict(color="black", lw=2))
    bracket_Y(ax, 0.01,textoffset=-6, text="BERT-E", color='green', scaley=256, scalex= 5/sum(scale), pos=[190.5/sum(scale),-1], linekw=dict(color="black", lw=2))
    bracket_Y(ax, 0.01,textoffset=-5, text="GloVe",  color='green', scaley=256, scalex= 5/sum(scale), pos=[196.5/sum(scale),-1], linekw=dict(color="black", lw=2))
    ax.text(55,-17,"Encoded Feature Space", size=108)
    ax.text(-30,145,"Decoded Feature Space", size=108, rotation=90)
    pos1 = ax.get_position().bounds




def pretty_print_mds(q1, l1, fname, writer='imagemagick'):

    def plot_gradient(q1, start, end, color, alpha_range, labels2, texts1=[], offset=0):
        alphas = np.linspace(alpha_range[0], alpha_range[1], end - start)
        for ei, i in enumerate(range(start, end - 1)):
            # Scale by l1
            plt.plot([q1[i, -1] * l1[-1], q1[i + 1, -1] * l1[-1]], [q1[i, -2] * l1[-2], q1[i + 1, -2] * l1[-2]], \
                     [q1[i, -3] * l1[-3], q1[i + 1, -3] * l1[-3]], 'o-', color=color, alpha=alphas[ei], lw=2, ms=3)
        return texts1

    ax = Axes3D(plt.figure(figsize=(8,8)))
    ax.set_xlim3d(-4, 4)
    ax.set_ylim3d(-4, 4)
    ax.set_zlim3d(-4, 4)

    colors = dict(semantic='blue',
                  syntactic='red',
                  embedding='green',
                  lowlevel='purple',
                  other='black',
                  lm='orange',
                  lm2='brown',
                  lm_out="black",
                  mt='pink',
                  mt2='indigo',
                  bi_lm='yellow',
                  lm3='cyan',
                  bi_lm2="lime")

    labels = np.array(['GloVe', "BERT-E", "FLAIR", 'POS',  'CHUNK', "NER", 'FRAME', "GloVe Next Word",
             "", "", "", "", "", "", "GPT2-Small", "", "", "", "", "",
            "", "", "", "", "", "GPT2-Medium", "", "", "", "", "", "",
            "", "", "", "", "", "", "", "", "", "", "", "",
            "", "", "", "", "", "", "Transformer-XL", "", "", "", "", "", "", "", "", "", "", "",
            "", "", "", "", "", "BERT", "", "", "", "", "", "", "",
            "", "", "", "", "", "", "ALBERT", "", "", "", "", "", "",
            "", "", "", "Eng → Zh", "", "",
            "", "", "", "Eng → De", "", ""])



    plot_gradient(q1, 8, 20, 'black', (0,0.7), labels)
    texts = plot_gradient(q1, 8, 20, 'orange', (0.7,0.7), labels, offset=2, texts1=[])

    plot_gradient(q1, 20, 44, 'black', (0,0.7), labels, texts1=[])
    texts = plot_gradient(q1, 20, 44, 'coral', (0.7,0.7), labels, texts, offset=3)

    plot_gradient(q1, 44, 62,"navy", (0,1), labels, texts1=[])
    texts = plot_gradient(q1, 44, 62, "fuchsia", (0.7,0.7), labels, texts, offset=3)

    plot_gradient(q1, 62, 75, "black", (0,1), labels, texts1=[])
    texts = plot_gradient(q1, 62, 75, "cadetblue", (0.5,0.5), labels, texts)

    plot_gradient(q1, 75, 88, "black", (0.3,1), labels, texts1=[])
    texts = plot_gradient(q1, 75, 88, "gray", (1,0), labels, texts, offset=5)

    plot_gradient(q1, 88, 94,"black", (0,1), labels, texts1=[])
    texts = plot_gradient(q1, 88, 94,"mediumvioletred", (0.7, 0.7), labels, texts, offset=9)

    plot_gradient(q1, 94, 100, 'green', (0,1), labels, texts1=[])
    texts = plot_gradient(q1, 94, 100, 'olive', (1,0), labels, texts)

    colors3d = ["green", "green", "green", "red", "red", "blue", "blue", "black"] + ["orange"] * 12 + ["coral"] * 24 + ["fuchsia"] * 18 + ["cadetblue"] * 13 + ["gray"] * 13 + ["mediumvioletred"] * 6 + ["olive"] * 6

    for qx, qy, qz, ll, ei in zip(q1[:,-1]*l1[-1], q1[:,-2]*l1[-2], q1[:,-3]*l1[-3], labels, list(range(100))):
        ax.plot([qx], [qy], [qz], 'o', color=colors3d[ei])
        ax.text(qx, qy, qz, ll, color=colors3d[ei], weight="bold")

    ax.view_init(45, 0)

    def animfxn(f):
        ax.view_init(25, f)
        return []

    anim = animation.FuncAnimation(plt.gcf(), animfxn, #init_func=lambda:None,
                                   frames=360, interval=20, blit=True)
    print("Writing gif...")
    anim.save(fname, writer=writer, fps=60)