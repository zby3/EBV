import staintools
import glob
from PIL import Image
import shutil
myinf1="/work/07034/byz/maverick2/EBV/Ref.png"
target = staintools.read_image(myinf1)
normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(target)
mydir='/work/07034/byz/maverick2/EBV/TCGA_STAD_norma_tiles/*.png'
myinf2=glob.glob(mydir)
nordir="/work/07034/byz/maverick2/EBV/TCGA_STAD_normalized"
processed="/work/07034/byz/maverick2/EBV/TCGA_STAD_norm_finished"
unprocessed="/work/07034/byz/maverick2/EBV/TCGA_STAD_norm_failed"
for i in myinf2:
    try:
        to_transform = staintools.read_image(i)
        transformed = normalizer.transform(to_transform)
        Im = Image.fromarray(transformed)
        Im = Im.resize((224, 224))
        outf = '/'.join([nordir, i[52:]])
        Im.save(outf, "PNG", quality=100)
        outfile = '/'.join([processed, i[52:]])
        shutil.move(i, outfile)
        print(i[52:], "is normalized")
    except:
        outfile = '/'.join([unprocessed, i[52:]])
        shutil.move(i, outfile)
        print(i[52:], "is failed to normalized")






