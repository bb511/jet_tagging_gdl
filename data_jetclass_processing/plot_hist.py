import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

class StreamingHist:
    def __init__(self,
                 data_classes: list[str] = ["QCD", "H → b¯b", "H → c¯c", "H → gg", "H → 4q",
                                            "H → ℓνqq′", "t → bqq′", "t → bℓν", "W → qq′", "Z → q¯q"],
                 feature_labels: list[str] = ["px", "py", "pz", "pt", "eta", "phi", "deta", "dphi"],
                 nb_constituents: bool = True,
                 bins: int = 60):
        self.classes = data_classes
        self.features = feature_labels
        self.nc = len(data_classes)
        self.nf = len(feature_labels)
        self.bins = bins
        self.edges: list[np.ndarray] = [None]*self.nf
        self.nb_constituents = nb_constituents
        self.nb_constituents_edges = np.linspace(0, 128, 100) if nb_constituents else None
        self.counts = None            # shape (nc, nf, bins)
        self.nb_constituents_counts = None  # shape (nc, bins)

    def _init_edges(self, x_chunk: np.ndarray):
        # x_chunk: (N, P, F) → min/max over all events+particles per feature
        lo = x_chunk.min(axis=(0,1))
        hi = x_chunk.max(axis=(0,1))
        self.edges = [
            np.linspace(lo[f], hi[f], self.bins+1)
            for f in range(self.nf)
        ]
        self.counts = np.zeros((self.nc, self.nf, self.bins), float)
        if self.nb_constituents:
            self.nb_constituents_counts = np.zeros((self.nc, len(self.nb_constituents_edges)-1), float)

    def _expand_edges(self, fi: int, new_lo: float, new_hi: float):
        old = self.edges[fi]
        lo, hi = min(old[0], new_lo), max(old[-1], new_hi)
        new_edges = np.linspace(lo, hi, self.bins+1)
        centers = 0.5*(old[:-1] + old[1:])
        # rebin old counts into new_edges
        for ci in range(self.nc):
            self.counts[ci, fi], _ = np.histogram(
                centers,
                bins=new_edges,
                weights=self.counts[ci, fi]
            )
        self.edges[fi] = new_edges

    def add_batch(self, x_chunk: np.ndarray, y_chunk: np.ndarray):
        # x_chunk: (N, P, F), y_chunk: (N,) or (N, nc)
        if self.edges[0] is None:
            self._init_edges(x_chunk)

        # get per-event labels
        labels = y_chunk.argmax(1) if y_chunk.ndim>1 else y_chunk
        for ci in range(self.nc):
            mask = (labels == ci)
            n_events = mask.sum()
            
            if n_events == 0:
                continue

            # flatten all particles for each feature
            for fi in range(self.nf):
                vals = x_chunk[mask, :, fi].reshape(-1)
                mn, mx = vals.min(), vals.max()
                e = self.edges[fi]
                if mn < e[0] or mx > e[-1]:
                    self._expand_edges(fi, mn, mx)
                    e = self.edges[fi]
                c, _ = np.histogram(vals, bins=e)
                self.counts[ci, fi] += c
                
            if self.nb_constituents:
                vals = x_chunk[mask, :, :]
                nb_constituents = (np.abs(vals).sum(axis=2) != 0).sum(axis=1)
                c, _ = np.histogram(nb_constituents, bins=self.nb_constituents_edges)
                self.nb_constituents_counts[ci] += c

    def plot(self,
             outdir: Path|None,
             split: str,
             color_map: dict[str,str]= {
                "QCD": "#7EB6E0",       # light blue
                "H → b¯b": "#822F88",   # dark purple
                "H → c¯c": "#E68CB7",   # pink
                "H → gg": "#B3B3B3",    # grey
                "H → 4q": "#F5B700",    # orange
                "H → ℓνqq′": "#E63946", # red
                "t → bqq′": "#52B788",  # green
                "t → bℓν": "#A8D5BA",   # light green
                "W → qq′": "#B6D6EB",   # pale blue
                "Z → q¯q": "#6F4DBF"    # indigo blue
                                        }):
        centers = [0.5*(e[:-1]+e[1:]) for e in self.edges]
        widths = [e[1]-e[0] for e in self.edges]
        if self.nb_constituents:
            centers_nb_constituents = 0.5*(self.nb_constituents_edges[:-1]+self.nb_constituents_edges[1:])
            widths_nb_constituents = self.nb_constituents_edges[1]-self.nb_constituents_edges[0]

        cols = 3
        rows = int(np.ceil(self.nf+1/cols)) if self.nb_constituents else int(np.ceil((self.nf)/cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols*6, rows*5),
                                 squeeze=False)
        axes = axes.flatten()

        for fi, feat in enumerate(self.features):
            ax = axes[fi]
            for ci, cls in enumerate(self.classes):
                dens = self.counts[ci, fi] / (self.counts[ci, fi].sum() * widths[fi])
                ax.step(centers[fi], dens,
                        where='mid',
                        lw=2,
                        label=cls,
                        color=color_map[cls],
                        alpha=0.7)
            ax.set_xlabel(feat)
            ax.set_yscale('log')
            ax.legend(fontsize=9)
        
        ax = axes[self.nf]
        for ci, cls in enumerate(self.classes):
            dens = self.nb_constituents_counts[ci] / (self.nb_constituents_counts[ci].sum() * widths_nb_constituents)
            ax.step(centers_nb_constituents, dens,
                    where='mid',
                    lw=2,
                    label=cls,
                    color=color_map[cls],
                    alpha=0.7)
        ax.set_xlabel("Number of constituents")
        ax.set_yscale('log')
        ax.legend(fontsize=9)

        if self.nb_constituents:
            for ax in axes[self.nf+1:]:
                fig.delaxes(ax)
        else:
            for ax in axes[self.nf:]:
                fig.delaxes(ax)
                
        fig.tight_layout()

        if outdir:
            p = outdir/f"jetclass_all_features_{split}.pdf"
            fig.savefig(p, bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()
        plt.close(fig)
        
def main():
    
    data_dir = Path("/cluster/scratch/sstelea/JetClass/all_feats_train_restrained")  
    x_files = sorted(data_dir.glob("x_*.npy"))
    y_files = sorted(data_dir.glob("y_*.npy"))

    # Match on the full identifier after 'x_' or 'y_'
    x_map = {f.name[2:]: f for f in x_files}  # 'x_' is 2 chars
    y_map = {f.name[2:]: f for f in y_files}  # 'y_' is 2 chars

    common_keys = sorted(set(x_map.keys()) & set(y_map.keys()))
    print(f"Found {len(common_keys)} matched pairs")

    hist = StreamingHist()
    
    for key in common_keys:
        print(f"Processing {key}")
        x = np.load(x_map[key])
        x = np.swapaxes(x, 1, 2)
        y = np.load(y_map[key])
        hist.add_batch(x, y)
    
    outdir = Path("dataplots_jetclass")
    hist.plot(outdir=outdir, split="train_restrained")
        
if __name__ == "__main__":
    main()
    