import openslide
import numpy as np
import os
import glob
import shutil
import pandas as pd
from tqdm import tqdm

# load patient ID with EBV data
myinf = "/byz/EBV/Cao2016_SciRep_SupTab1_TCGA_Virus.txt"
info = pd.read_csv(myinf, sep="\t")
info = info[info["Cancer"] == "STAD"]
ID = set(df['Sample'].apply(lambda x: x[0:12]))

# set tile size and threshold for background exclusion
tile_size = 512
bg_th = 220
max_bg_frac = 0.5

# cut each large svs files into small tiles
tilesdir = "/byz/EBV/TCGA_STAD_tiles"
svsdir = '/byz/EBV/TCGA_STAD_svs/*.svs'
myinf1 = glob.glob(svsdir)

for i in tqdm(myinf1):
    pat = i.split('/')[-1][0:12]  # get pat ID from each svs filename
    if pat in ID:
        slide = openslide.OpenSlide(i)
        try:
            downsample = int(slide.properties['aperio.AppMag']) / 20
        except:
            f"{i.split('/')[-1][0:16]} is corrupted"
            slide.close()
        else:
            L = int(downsample * tile_size)
            Xdims = np.floor((np.array(slide.dimensions) - 1) / L).astype(int)
            num_tiles = np.prod(Xdims)
            for m in range(Xdims[0]):
                for n in range(Xdims[1]):
                    tile = slide.read_region((m * L, n * L), 0, (L, L))
                    tile = tile.convert('RGB')  # remove alpha channel
                    tile = tile.resize((tile_size, tile_size))
                    if (np.array(tile).min(axis=2) >= bg_th).mean() <= max_bg_frac:  # remove tiles that are background
                        outfile = '/'.join([tilesdir, '%s_tile_%d_%d.png' % (i.split('/')[-1][0:16], m, n)])
                        tile.save(outfile, "PNG", quality=100)
            slide.close()
