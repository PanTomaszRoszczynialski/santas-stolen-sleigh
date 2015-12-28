# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:43:24 2015

@author: tomek
"""

import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine
import csv
import random

north_pole = (90,0)
sleigh_mass = 10.0
total_wrw=0

def readData(ListFile):
    f=open(ListFile,'rb')
    reader = csv.reader(f)
    GiftList = list()
    for raw in list(reader)[1:]:
        GiftList.append(map(float,raw[1:]))
    return GiftList    
    
def writeSolution(GiftList,Trips,file_name):
    with open(file_name,'w') as f :
        wrt = csv.writer(f,delimiter=',')
        wrt.writerow(['GiftId','TripId'])
        for tr in range(len(Trips)):
            for g in Trips[tr] : 
                wrt.writerow([str(g),str(tr)])
    

def euclid_dist(x,y):
    return math.sqrt((x[1]-y[1])**2+(x[2]-y[2])**2)

def boltzmannProb(dif,T):
    return np.exp(-dif/T)
    
def initializeTrips(GiftList):
    Trips = []
    for it in range(len(GiftList)):
        Trips.append([it])
        GiftList[it].append(it)
        GiftList[it].append(0)
    return Trips


def trip_wrw(trip,Trips,GiftList,distance=haversine):
  #  print trip
    route = Trips[trip]
    if len(route)==0 :
        return 0
    places = [(GiftList[x][0],GiftList[x][1]) for x in route]
    weights = [GiftList[x][2] for x in Trips[trip]]
    total_mass = sum(weights)+sleigh_mass
    tot=total_mass*distance(north_pole,places[0])
    for it in range(len(places)-1):
        total_mass-=weights[it]
        tot+=total_mass*distance(places[it],places[it+1])
    
    tot+=sleigh_mass*haversine(places[len(places)-1],north_pole)
    return tot
    
def initalize_wrws(Trips,GiftList,distance):
    wrws = []
    for it in range(len(Trips)):
        
        wrws.append(trip_wrw(it,Trips,GiftList,distance))
    return wrws
    
def totalWRW(Trips,GiftList,distance=haversine):
    s=0
    for it in range(len(Trips)):
        s+=trip_wrw(it, Trips, GiftList,distance)
    return s
    
        
def mergeTrips(trip1,trip2,Trips):
    Trips[trip1]+=Trips[trip2]
    Trips[trip2]=list()
        
def updateAfterMerge(trip1,trip2,GiftList):
    if trip1==trip2:
        raise Exception(0)
    s=len(Trips[trip1])
    
    for it in Trips[trip2]:
        GiftList[it][3]=trip1
        GiftList[it][4]=s
        s+=1
           
        
def checkIfMerge(trip1,trip2,Trips,GiftList,distance=haversine):
    l=list([list(Trips[trip1]),list(Trips[trip2])])
    l_c = list(l)
    Ei = trip_wrw(0,l,GiftList,distance)+trip_wrw(1,l,GiftList,distance)
    #print l
    mergeTrips(0,1,l)
    #print 'Jestem w checku',l
    Ef = trip_wrw(0,l,GiftList,distance)
    
    return Ef-Ei
    
def permuteGiftsInTrip(new_route,GiftList,trip,Trips):
    route = Trips[trip]
    #if trip!=GiftList[new_route[0]][3]:
    #    raise Exception(0)
    if len(new_route)!=len(route) :
        raise Exception(0)
    route = new_route

def updateAfterPermutation(new_route,GiftList,Trips):
    pass
    
def checkIfPermute(new_route,GiftList,trip,Trips):
    l=[list(Trips[trip])]
    E_i = trip_wrw(trip,Trips,GiftList)
    permuteGiftsInTrip(new_route,GiftList,0,l)
    E_f = trip_wrw(0,l,GiftList)
    
    return E_f-E_i
        

def avarageDifference(GiftList,Trips,samples):
    s_plus=0
    s_minus=0
    c_minus=0
    c_plus=0
    N = len(GiftList)

    for it in range(samples):
        id1 = random.randrange(0,N)
        id2 = random.randrange(0,N)
        if id1==id2 :
            continue
        trip1 = GiftList[id1][3]
        trip2 = GiftList[id2][3]
        if trip1==trip2:
            continue
        dif = checkIfMerge(trip1,trip2,Trips,GiftList)
        if dif <= 0 :
            c_minus-=1
        else :
            c_plus+=1
            s_plus+=dif
    result = [c_minus,c_plus,s_plus/N]
    return result

def optimize1(T_start,iterations,GiftList,Trips,prob=boltzmannProb):
    t_plus = 0
    t_minus=0
    wrw = totalWRW(Trips,GiftList)
    epsilon = (T_start+100.0)/float(iterations)
    T=T_start+epsilon
    N=len(GiftList)
    for it in range(iterations):
        T-=epsilon
        if T<=0 :
            return wrw
        print 'Temperature:',T
        id1 = random.randrange(0,N)
        id2 = random.randrange(0,N)
        gift1 = GiftList[id1]
        gift2 = GiftList[id2]
        trip1,trip2 = gift1[3],gift2[3]
        if trip1==trip2:
            continue
 
        #print 'Trips',trip1,trip2,'chosen.'
        dif = checkIfMerge(trip1,trip2,Trips,GiftList)
        #print  'Their difference: ',dif
        if dif < 0 :
            if len(Trips[trip1])+len(Trips[trip2])>=0.0 :
                #print 'negative, merging'
                updateAfterMerge(trip1,trip2,GiftList)
                mergeTrips(trip1,trip2,Trips)
                wrw+=dif
                t_minus+=dif
                
                #print Trips

        else :
            r = random.random()
            #print 'Boltzmann:', prob(dif,T),'random number:', r
            if prob(dif,T)>r:
                if len(Trips[trip1])+len(Trips[trip2])>=0.0:
                    #print 'merging'
                    updateAfterMerge(trip1,trip2,GiftList)
                    mergeTrips(trip1,trip2,Trips)
                    wrw+=dif
                    t_plus+=dif
                    #print Trip
    return wrw




GiftList = readData('data/gifts.csv')[1:10000]
Trips = initializeTrips(GiftList)
#r = avarageDifference(GiftList,Trips,10000)

before = totalWRW(Trips,GiftList)

opt=optimize1(15000.0,100000,GiftList,Trips)
print 'Initial WRW', before
print 'After optimization', opt
s=0
n=0
for it in Trips:
    s+=len(it)
    if len(it)!=0:
        n+=1
print 'Avarage trip length:',float(s)/float(n)
writeSolution(GiftList,Trips,'dupa1')



#print w 





#writeSolution(GiftList,'dupa1')









        
