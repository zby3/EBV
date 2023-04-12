import os
import staintools
import glob
from PIL import Image
import shutil
from tqdm import tqdm
import argparse

parser.add_argument('--input', type=str,help='input path')
parser.add_argument('--ref', type=str,help='Reference image path')
parser.add_argument('--output', type=str,help='output path')
args = parser.parse_args()

myinf1=args.ref
target = staintools.read_image(myinf1)
normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(target)
mydir=os.path.join(args.input,'*.png')
myinf2=glob.glob(mydir)
nordir=args.output
for i in tqdm(myinf2):
    try:
        to_transform = staintools.read_image(i)
        transformed = normalizer.transform(to_transform)
        Im = Image.fromarray(transformed)
        Im = Im.resize((224, 224))
        outf = os.path.join(nordir, i.split('/')[-1])
        Im.save(outf, "PNG", quality=100)
    except:
        print(i.split('/')[-1], "is failed to normalized")






