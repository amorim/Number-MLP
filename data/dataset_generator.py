import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def number_to_index(pool):
    mapao = {}
    for i in range(len(pool)):
        mapao[pool[i]] = i
    return mapao


class Generator:
    def __init__(self, ntrain, ntest, number_pool, size, dont_repeat=False):
        self.ntrain = ntrain
        self.ntest = ntest
        self.tot = ntrain + ntest
        self.number_pool = number_pool
        self.size = size
        self.dont_repeat = dont_repeat

    def gen(self, report_progress=False):
        mapao = number_to_index(self.number_pool)
        dataset = []
        for i in range(self.tot):
            if report_progress and i % 1000 == 0:
                print('Dataset generation is in progress: ' + str((float(i) / self.tot) * 100) + '%')
            numba = self.number_pool[random.randint(0, len(self.number_pool) - 1) if not self.dont_repeat else i]
            string = str(numba)
            img = Image.new('RGBA', (self.size, self.size), color=(255, 255, 255, 0))
            font = ImageFont.truetype(
                'C:/Windows/Fonts/arial.ttf' if random.random() < 0.5 else 'C:/Windows/Fonts/arialbd.ttf', 18)
            d = ImageDraw.Draw(img)
            w, h = d.textsize(string, font=font)
            d.text(((self.size - w) / 2, (self.size - h - font.getoffset(string)[1]) / 2), string, font=font,
                   fill=(0, 0, 0))
            rot = img.rotate(random.randint(-45, 45)).transform(img.size, Image.AFFINE, (
            1, 0, random.randint(-7, 7), 0, 1, random.randint(-7, 7)))
            fff = Image.new('RGBA', img.size, (255,) * 4)
            out = Image.composite(rot, fff, rot)
            out = out.convert('L')
            #out.save('case' + str(i) + '.' + string + '.bmp')
            pix = out.load()
            arr = np.empty((self.size * self.size, 1), float)
            for j in range(self.size):
                for k in range(self.size):
                    arr[j * self.size + k] = float(abs(pix[k, j] - 255)) / 255.
            if i < self.ntrain:
                k = np.zeros((len(self.number_pool), 1), float)
                k[mapao[numba]] = 1.
            else:
                k = numba
            dataset.append((arr, k))
        return dataset