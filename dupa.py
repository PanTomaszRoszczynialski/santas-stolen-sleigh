import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine
import csv
import random

north_pole = (90,0)
sleigh_mass = 10.0

def distance(destination, origin = north_pole):
    """ Calculates distance from the north pole if
        one argument is provided """
    return haversine(origin, destination)

def read_data(ListFile):
    GiftList = []
    f = open(ListFile, 'rb')
    reader = csv.reader(f)

    # Skip csv header
    reader.next()

    for row in reader:
        # Keep gift coordinates and ommit gift-id from the csv file
        gift = [float(val) for val in row[1:]]
        GiftList.append(gift)

    return GiftList

def write_solution(GiftList, Trips, file_name):
    with open(file_name, 'w') as f :
        wrt = csv.writer(f, delimiter=',')
        wrt.writerow(['GiftId', 'TripId'])
        for trip in range(len(Trips)):
            for gift in Trips[trip] :
                wrt.writerow([str(gift), str(trip)])

def euclid_dist(x, y):
    """ Unused """
    return math.sqrt((x[1]-y[1])**2+(x[2]-y[2])**2)

def boltzmann_distribution(dif, T):
    return np.exp(-dif/T)

def init_trips(GiftList):
    Trips = []
    for it in range(len(GiftList)):
        Trips.append([it])

        # Trip id (initially every gift is in separate trip)
        GiftList[it].append(it)

        # 0-based position in trip's queue
        GiftList[it].append(0)

    return Trips

#List of total trip masses can be useful in future     
def init_trips1(GiftList):
    N = len(GiftList)
    l = list(np.random.permutation(N))
    result = []
    trip = 0
    mass = 0
    result.append([])
    for it in l:
        #print mass, result
        mass += GiftList[it][2]
        if mass <= 1000:
            result[trip].append(it)
            GiftList[it].append(trip)
        else:
            trip += 1
            mass = 0
            result.append([])
            result[trip].append(it)
            GiftList[it].append(trip)

    return result

def init_masses(Trips):
    Masses = []
    for it in range(len(Trips)):
        Masses.append(GiftList[it][2])

    return Masses

def trip_wrw(trip, GiftList):
    """ trip is a list of gifts to be delivered in one session """
    if len(trip) is 0:
        return 0

    places = [(GiftList[x][0], GiftList[x][1]) for x in trip]
    weights = [GiftList[x][2] for x in trip]
    total_mass = sum(weights) + sleigh_mass

    tot = total_mass * distance(places[0])
    for it in range(len(places)-1):
        total_mass -= weights[it]
        tot += total_mass * distance(places[it], places[it+1])

    # Get back to the pole, bitch
    tot += sleigh_mass * distance(places[-1])

    return tot

def initalize_wrws(Trips, GiftList):
    wrws = []
    for trip in Trips:
        wrws.append(trip_wrw(trip, GiftList))

    return wrws

def total_WRW(Trips, GiftList):
    s = 0
    for trip in Trips:
        s += trip_wrw(trip, GiftList)

    return s

def merge_trips(trip1, trip2, Trips):
    """ dafuq """
    Trips[trip1] += Trips[trip2]
    Trips[trip2] = list()

def update_after_merge(trip1, trip2, GiftList,Masses):
    """ no idea what's going on in here """
    if trip1 == trip2:
        print 'wtf, this doesn\'t happen'
        raise Exception(0)

    s = len(Trips[trip1])

    for it in Trips[trip2]:
        GiftList[it][3] = trip1
        GiftList[it][4] = s
        s += 1

    Masses[trip1] += Masses[trip2]
    Masses[trip2] = 0

def check_if_merge(trip1, trip2, Trips, GiftList):
    """ This function seems to be important """
    # Some serious resource managament issue is 
    # clearly visible here, there is no pretty
    # way to make sure we get copies from Trips
    # and not references ???
    # l = [Trips[trip1][:], Trips[trip2][:]]
    local_trips = [list(Trips[trip1]), list(Trips[trip2])]

    Ei = trip_wrw(local_trips[0], GiftList) +\
         trip_wrw(local_trips[1], GiftList)

    # no kurwa
    merge_trips(0, 1, local_trips)
    # XXX Karny kutas za jezykowy promiskuityzm
    #print 'Jestem w checku',l
    Ef = trip_wrw(local_trips[0], GiftList)

    return Ef - Ei

def split_trips(trip,Trips,place):
    #place = GiftList[gift][4]
    Trips.append(Trips[trip][place:])
    Trips[trip] = Trips[trip][:place]

def check_if_split(trip, Trips, place, GiftList) :
    E_i = trip_wrw(Trips[trip], GiftList)

    E_f = trip_wrw(Trips[trip][:place], GiftList) +\
          trip_wrw(Trips[trip][place:], GiftList)

    return E_f-E_i

def update_after_split(trip,Trips,place,GiftList):
    for it in Trips[trip][place:]:
        GiftList[it][3] = len(Trips)
        Masses[trip] = sum([GiftList[x][3] for x in Trips[trip][:place]])
        Masses.append(sum([GiftList[x][3] for x in Trips[trip][place:]]))


def permute_gifts_in_trip(new_route, GiftList, trip, Trips):
    """ Used only in unused function """
    route = Trips[trip]
    #if trip!=GiftList[new_route[0]][3]:
    #    raise Exception(0)
    if len(new_route) is not len(route) :
        raise Exception(0)
    route = new_route

def update_after_permutation(new_route, GiftList, trip, Trips):
    """ Unused """
    for it in range(len(new_route)):
        GiftList[new_route[it]][4]=it

def check_if_permute(new_route, GiftList, trip, Trips):
    """ Unused """
    l = [list(Trips[trip])]
    E_i = trip_wrw(Trips[trip], GiftList)
    permute_gifts_in_trip(new_route, GiftList, 0, l)
    E_f = trip_wrw(l[0], GiftList)

    return E_f-E_i

def avarage_difference(GiftList, Trips, samples):
    """ Unused """
    s_plus  = 0
    s_minus = 0
    c_minus = 0
    c_plus  = 0
    N = len(GiftList)

    for it in range(samples):
        id1 = random.randrange(0,N)
        id2 = random.randrange(0,N)
        if id1==id2 :
            continue
        trip1 = GiftList[id1][3]
        trip2 = GiftList[id2][3]
        if trip1 == trip2:
            continue

        dif = check_if_merge(trip1, trip2, Trips, GiftList)

        if dif <= 0 :
            c_minus-=1
        else:
            c_plus+=1
            s_plus+=dif

    result = [c_minus, c_plus, s_plus/N]

    return result

def optimize1(T_start, iterations,
              GiftList, Trips, Masses,
              prob = boltzmann_distribution):

    # What are those t_ for?
    t_plus  = 0
    t_minus = 0

    wrw = total_WRW(Trips, GiftList)
    epsilon = (T_start + 100.0)/float(iterations)
    #przepraszam za jezykowy "promiskuityzm" ale do JASNEJ KURWY czemu 
    #temperatura dochodzi ponizej zera przy takiej  inicjalizacji??????
    #Nie ogarniam    

    T = T_start + epsilon
    N = len(GiftList)

    # Keep track of improvements for diagnostics
    scores = []

    for it in range(iterations):
        # Take temperature step
        T -= epsilon

        # Save current score
        scores.append(wrw)

        # Print debug information every 1000 steps
        if it%1000 is 0:
            debug_str = 'T: {0}, Iteration: {1}, Score: {2}'
            print debug_str.format(T, it, wrw)

        # Procedure usually ends here
        if T <= 0:
            debug_str = 'T: {0}, Iteration: {1}, Score: {2}'
            print debug_str.format(T, it, wrw)
            return scores

        # Take 2 random gifts
        id1 = random.randrange(0, N)
        id2 = random.randrange(0, N)
        gift1 = GiftList[id1]
        gift2 = GiftList[id2]
        trip1, trip2 = gift1[3], gift2[3]

        if trip1 == trip2:
            continue

        # Check if it is beneficial to take the second gift
        # on the trip carrying the first one ???
        dif = check_if_merge(trip1, trip2, Trips, GiftList)

        if dif < 0:
            if Masses[trip1] + Masses[trip2] <= 1000:
                #print 'negative, merging'
                update_after_merge(trip1, trip2, GiftList,Masses)
                merge_trips(trip1, trip2, Trips)
                wrw += dif
                t_minus += dif
                #print Trips
        else:
            r = random.random()
            #print 'Boltzmann:', prob(dif,T),'random number:', r
            if prob(dif, T) > r:
                if Masses[trip1] + Masses[trip2] >= 0.0:
                    #print 'merging'
                    update_after_merge(trip1, trip2, GiftList,Masses)
                    merge_trips(trip1, trip2, Trips)
                    wrw += dif
                    t_plus += dif
                    #print Trip

    # I don't know what those are representing
    # but I trye to print them anyway
    print 't+: {0}, t-:{1}'.format(t_plus, t_minus)

    return scores

def optimize2(T_start, iterations,
              GiftList, Trips, Masses,
              prob = boltzmann_distribution):

    # What are those t_ for?
    t_plus  = 0
    t_minus = 0

    wrw = total_WRW(Trips, GiftList)
    epsilon = (T_start + 100.0)/float(iterations)
    #przepraszam za jezykowy "promiskuityzm" ale do JASNEJ KURWY czemu 
    #temperatura dochodzi ponizej zera przy takiej  inicjalizacji??????
    #Nie ogarniam    

    T = T_start + epsilon
    N = len(GiftList)
    N_T = len(Trips)

    # Keep track of improvements for diagnostics
    scores = []

    for it in range(iterations):
        # Take temperature step
        T -= epsilon

        # Save current score
        scores.append(wrw)

        # Print debug information every 1000 steps
        if it%1000 is 0:
            debug_str = 'T: {0}, Iteration: {1}, Score: {2}'
            print debug_str.format(T, it, wrw)

        # Procedure usually ends here
        if T <= 0:
            debug_str = 'T: {0}, Iteration: {1}, Score: {2}'
            print debug_str.format(T, it, wrw)
            return scores

        # Take random trip
        trip = random.randrange(0,len(Trips))
        if len(Trips[trip]) == 1 or len(Trips[trip]) == 0 :
            continue
        #take random place in the trip
        place = random.randrange(0,len(Trips[trip]))
        # Check if it is beneficial to take the second gift
        # on the trip carrying the first one ???
        dif = check_if_split(trip,Trips,place,GiftList)
        
        if dif < 0:
                #print 'negative, merging'
            update_after_split(trip,Trips,place,GiftList)
            split_trips(trip,Trips,place)
            wrw += dif
            t_minus += dif
                #print Tri
        else:
            r = random.random()
            #print 'Boltzmann:', prob(dif,T),'random number:', r
            if prob(dif, T) > r:
                update_after_split(trip,Trips,place,GiftList)
                split_trips(trip, Trips, place)
                wrw += dif
                t_minus += dif
                    #print 'merging'

                    #print Trip

    # I don't know what those are representing
    # but I trye to print them anyway
    print 't+: {0}, t-:{1}'.format(t_plus, t_minus)

    return scores


if __name__ == "__main__":


    GiftList = read_data('data/gifts.csv')[0:10000]
    Trips = init_trips(GiftList)
    Masses=init_masses(Trips)
    #r = avarage_difference(GiftList,Trips,10000)

    before = total_WRW(Trips, GiftList)

    iterations = 2
    results = optimize1(15000.0, iterations, GiftList, Trips,Masses)
    after = results[-1]

    print 'Initial WRW', before
    print 'After optimization', after

    plt.plot(results)
    plt.show()

    # This seems to be bullshit
    s = 0
    n = 0

    for it in Trips:
        s += len(it)
        if len(it) is not 0:
            n += 1

    print 'Avarage trip length:', float(s)/float(n)

    write_solution(GiftList, Trips, 'local_solution.csv')
   
    iterations1=100
    GiftList = read_data('data/gifts.csv')[0:10000]
    Trips = init_trips1(GiftList)
    
    before = total_WRW(Trips, GiftList)

    iterations = int(4e4)
    results = optimize2(15000.0, iterations, GiftList, Trips,Masses)
    after = results[-1]
    plt.plot(results)
    plt.show()

    print 'Initial WRW', before
    print 'After optimization', after
    
    s = 0
    n = 0

    for it in Trips:
        s += len(it)
        if len(it) is not 0:
            n += 1

    print 'Avarage trip length:', float(s)/float(n)
    

    
    
    
    

