# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:14:52 2017

@author: rezaandriyunanto
"""

import Image
import time
import random
from colorsys import rgb_to_hsv
import glob



print time.strftime("%H:%M:%S")

   
def metric(pixel1,pixel2):
    x1, y1, h1, s1, v1, px1 = pixel1
    x2, y2, h2, s2, v2, px2 = pixel2
    #comparison = ((x2-x1)**2+(y2-y1)**2+(px2-px1)**2)**0.5
    weighted = 19.9
    comparison = ((x2-x1)**2+(y2-y1)**2+((h1*weighted)-(h2*weighted))**2+((s2*weighted)-(s1*weighted))**2+((v2*weighted)-(v1*weighted))**2)**0.5

    return comparison

def metric2(pixel1,pixel2):
    ans = 0
    for x,y in zip(pixel1,pixel2):
        ans += abs(x-y)
    comparison = (float(ans)/len(pixel1))
##    if int(comparison) % 2 == 0:
##        comparison *= 90
    return comparison

class Cluster(object):
    def __init__(self,initial_pixels):
        # initial pixels = a list of pixel-tuples
        self.pixels = initial_pixels
        self.centre = self.calculate_centre()

    def calculate_centre(self):
        length = float(len(self.pixels))
        xs, ys, hs, ss, vs, pixels = zip(*self.pixels)
        xs = sum(xs)/length
        ys = sum(ys)/length
        hs = sum(hs)/length
        ss = sum(ss)/length
        vs = sum(vs)/length
        
        return (xs,ys,hs,ss,vs,0)
        
    def recalculate_centre(self):
        self.centre = self.calculate_centre()
        
    def get_centre(self):
        return self.centre
        
    def get_pixels(self):
        return self.pixels
        
    def add_pixel(self,pixel):
        self.pixels.append(pixel)
        self.recalculate_centre()
        
    def add_pixels(self,pixels):
        for pixel in pixels:
            self.pixels.append(pixel)
        self.recalculate_centre()
    
    def add_cluster(self,other_cluster):
        self.pixels.extend(other_cluster.get_pixels())
        self.recalculate_centre()
        
    def __len__(self):
        return len(self.pixels)


class Blank_Cluster(Cluster):
    def __init__(self,initial_centre):
        # initial pixels = a list of pixel-tuples
        self.pixels = [(initial_centre[0], initial_centre[1],
                        initial_centre[2], initial_centre[3],
                        initial_centre[4], 0)]
        self.centre = initial_centre
        #self.iters = 0

    def add_cluster(self,other_cluster):
        self.pixels.extend(other_cluster.get_pixels())
        #self.iters += 1
        #if self.iters % update_rate == 0:
        #    self.recalculate_centre()

    def get_pixels(self):
        return self.pixels[1:]

        
k = 15        # number of clusters
limit = 280   # max number of iterations


for chosen_image in glob.glob():

    im = Image.open("sampledataset.jpg")
    
    mode = im.mode
##    while max(im.size) > 1200:
##        isx, isy = im.size
##        im = im.resize((isx/2,isy/2), Image.BICUBIC)
##    im.save("RESIZED.png")
    isx, isy = im.size
    loadobject = im.load()
    print im.size
    #print im.filename
##    log = open("log.txt","a+")
##    log.writelines(["K = " + str(k),"FILE = " + chosen_image])
##    log.write("\n")
    pixels = []

    print "INIT CLUSTER LIST"
    
    for x in xrange(isx):
        for y in xrange(isy):
            px1 = loadobject[x,y]
            #r,g,b = px1
            #h,s,v = rgb_to_hsv(r/255.0,g/255.0,b/255.0)
            h,s,v = px1
            pixels.append(Cluster([(x,y,float(h),float(s),float(v),px1)]))

    im = Image.new(mode,im.size)

    print "NUM POINTS: ", len(pixels)
    print ""

    centroids = []
    for x in range(k):
        index = random.randint(0,len(pixels))
        centroids.append(Blank_Cluster(pixels[index].get_centre()))


    random.shuffle(pixels)
    iters = 0


    dev = 10000
    sizes = []

    print "RUN K-MEANS"
    counter_ = 0
    while iters < 144 and dev > limit:
        counter_ = 0
        centroids = [Blank_Cluster(c.get_centre()) for c in centroids]
        print "  ITERATION", iters
        for pixel in pixels:
            comps = [metric(pixel.get_centre(),centroid.get_centre()) for centroid in centroids]
            index = comps.index(min(comps))
            centroids[index].add_cluster(pixel)
            counter_ += 1
            if counter_ % 5000 == 0:
                print counter_,
        print ""
        for c in centroids:
            c.recalculate_centre()
        sizes.append([len(c) for c in centroids])
        print "  CENTROIDS: "
        for b in [c.get_centre() for c in centroids]:
            print "    ", b
        
        
        if len(sizes) > 1:
            devs = [abs(f-g)**2 for f, g in zip(sizes[-1],sizes[-2])]
            dev = (sum(devs)/float(len(devs)))**0.5
            print "   ALL DEVS :", devs
            print "      DEV = ", dev
            log.write(str(dev) + ", ")
            print "      MAX DEV = ", max(devs)
            print ""
            
            '''
            if devs < limit:
                break
            '''
        print ""
        iters += 1
        random.shuffle(pixels)
        
        
        



    clusters = [a.get_pixels() for a in centroids]

    loadobject = im.load()

    for enum, clust in enumerate(clusters):
        xs, ys, hs, ss, vs, pixs = zip(*clust)
        centroid = centroids[enum].get_centre()
        cent_xy = centroid[0:2]
        ords = sorted(zip(xs,ys),key=lambda d: metric2(d,cent_xy))
        pixs = sorted(pixs)  #reverse = True)
    
        #pixs = sorted(pixs,key=lambda a: rgb_to_hsv(a[0]/255.0,a[1]/255.0,a[2]/255.0)[0])
        for coord, pixel in zip(ords,pixs):
            try:
            
                x, y = coord
                loadobject[x,y] = pixel
            except:
                print coord, pixel

    im.save("KMEANS" + str(k) + "_" + str(random.randint(100,4000)) + ".png")
    print time.strftime("%H:%M:%S")
    ##log.write("\n\n\n")
    ##log.close()