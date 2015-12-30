import numpy as np
import pandas as pd
from haversine import haversine

north_pole = (90, 0)
weight_limit = 1000
sleigh_weight = 10

# XXX Longitude is constant for trips from north pole south
# and it's a prominent example of badly named variable
# so we go to more intuitive theta and phi convention
# where theta goes from pole to pole
_TH = 'Longitude'
_PH = 'Latitude'

class Trip(object):
    """ Basic holder for santa trips """
    def __init__(self):
        """ Main constructor, initializes an empty trip """

        # ID of each gift, useful only for submission generation
        self.gifts = []

        # Mass
        self.weights = []

        # Location
        self.positions = []

    def create_trip(self, gifts):
        """ Creates trip from pandas.DataFrame object """

        # Sleigh udzwig assert FIXME no internet no translator
        if gifts.Weight.sum() > weight_limit:
            print 'this should never happen in Trip::create_trip'

        # Change pandas into simpler lists
        self.gifts = list(gifts.GiftId)
        self.weights = list(gifts.Weight)
        longs = list(gifts.Longitude)
        latds = list(gifts.Latitude)
        self.positions = [[long, latd] for long, latd in zip(longs,latds)]

        # FIXME hide this 
        print 'Created trip with: {0} wrw'.format(self.wrw())

    def full_weight(self):
        """ Returns mass of all presents and santas fat ass """
        return sleigh_weight + sum(self.weights)

    def wrw(self):
        """ Returns weighted reindeer weariness for this trip """
        prev_stop = north_pole
        prev_weight = self.full_weight()

        # Cumulate
        dist = 0.0
        for it in range(len(self.gifts)):
            # Current location
            c_location = self.positions[it]
            dist += distance(c_location, prev_stop)\
                    * prev_weight
            prev_stop = c_location
            prev_weight -= self.weights[it]

        # Add wrw of returning to the northpole (no presents)
        dist += distance(prev_stop) * sleigh_weight

        return dist

def distance(destination, origin = north_pole):
    """ Provide single position for to/from northpole trips """
    return haversine(origin, destination)

def read_gifts(path = "data/gifts.csv"):
    """ Read the file with gifts and return a pd.DataFrame object """
    # Don't know wha is the fillna() method for
    gifts = pd.read_csv(path).fillna(" ")

    return gifts

def show_gifts_info(gifts):
    """ Pandas usability showcase """

    # Gifts weights
    mean_mass = gifts.Weight.mean()
    mass_std  = gifts.Weight.std()
    mass_info = 'Mean value of mass: {0}, with standard deviation: {1}'
    print mass_info.format(mean_mass, mass_std)

    # Gifts positions
    min_long = gifts.Longitude.min()
    max_long = gifts.Longitude.max()
    min_lat = gifts.Latitude.min()
    max_lat = gifts.Latitude.max()
    print 'Latitude span: [{0}, {1}]'.format(min_lat, max_lat)
    print 'Longitude span: [{0}, {1}]'.format(min_long, max_long)

def find_on_meridian(gifts, long, sigma = 0.0):
    """ Finds trips with the same longitude with accuracy of sigma """

    # TODO account for periodicity of a sphere
    up_limit = long + sigma
    down_limit = long - sigma
    out = gifts.loc[(gifts[_TH] < up_limit) & (gifts[_TH] > down_limit)]

    return out
