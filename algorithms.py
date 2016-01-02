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

def distance(destination, origin = north_pole):
    """ Provide single position for to/from northpole trips """
    return haversine(origin, destination)

def read_gifts(path = "data/gifts.csv"):
    """ Read the file with gifts and return a pd.DataFrame object """
    # Don't know wha is the fillna() method for
    gifts = pd.read_csv(path, index_col=0).fillna(" ")

    return gifts

# Global gifts container
_GIFTS = read_gifts()

def gift_mass(gid):
    """ Returns weight of gift """
    return _GIFTS.loc[gid, 'Weight']

def gift_theta(gid):
    """ Returns longitude of gift """
    return _GIFTS.loc[gid, _TH]

def gift_phi(gid):
    """ Returns latitude of gift """
    return _GIFTS.loc[gid, _PH]

def gift_position(gid):
    """ Returns gift coordinates as accepted by the distance method """
    return gift_theta(gid), gift_phi(gid)

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
    """ Finds gifts with the same longitude with accuracy of sigma """

    # TODO account for periodicity of a sphere
    up_limit   = long + sigma
    down_limit = long - sigma
    out = gifts.loc[(gifts[_TH] < up_limit) & (gifts[_TH] > down_limit)]

    return out

class Trip(object):
    """ Basic holder for santa trips """
    def __init__(self):
        """ Main constructor, initializes an empty trip """

        # ID of each gift, useful only for submission generation
        self.gifts = []

        # Only summed mass is interesting
        self.mass = sleigh_weight

    # FIXME this should probably be in the main constructor 
    def create_trip(self, gifts):
        """ Creates trip from pandas.DataFrame object """

        # Sleigh capacity assert 
        if gifts.Weight.sum() > weight_limit:
            print 'this should never happen in Trip::create_trip'

        # Change pandas into simpler lists
        # pandas array is now indexed with GiftId
        self.gifts = list(gifts.index)

        self.mass = sleigh_weight + sum(gifts.Weight)

        # FIXME hide this 
        if False:
            print 'Created trip with: {0} wrw'.format(self.wrw())

    def full_weight(self):
        """ Returns mass of all presents and santas fat ass """
        return self.mass

    def wrw(self):
        """ Returns weighted reindeer weariness for this trip """
        prev_stop = north_pole
        prev_weight = self.full_weight()

        # Cumulate
        dist = 0.0
        for git in self.gifts:
            # Current location
            c_location = gift_position(git)
            dist += distance(c_location, prev_stop)\
                    * prev_weight
            prev_stop = c_location
            prev_weight -= gift_mass(git)

        # Add wrw of returning to the northpole (no presents)
        dist += distance(prev_stop) * sleigh_weight

        return dist

    def remove_gift(self, gid):
        """ Remove gift with GiftId, not local index in self.list """
        self.gifts.remove(gid)
        self.mass -= gift_mass(gid)

    def insert_gift(self, where, gid):
        """ Insert a gift """
        self.gifts.insert(where, gid)
        self.mass += gift_mass(gid)

    def swap(self, gift_id, other_gift):
        """ Switch with gift from outside this trip """
        pass

    def castling(self, first, second):
        """ Changing order of gifts within a trip """
        self.gifts[first], self.gifts[second] =\
        self.gifts[second], self.gifts[first]
        pass

if __name__ == '__main__':
    not_delivered = _GIFTS.copy()

    trips = []
    while len(not_delivered.index) is not 0:
        debug_str = 'Gifts left on the pole: {0}'
        debug_str += ' after {1} trips.'
        if len(trips)%20 == 0:
            print debug_str.format(len(not_delivered.index),
                                   len(trips))
        # Take random gift's longitude
        # sample() gets random row, iloc[0] enables label indexing
        long = not_delivered.sample().iloc[0][_TH]

        # Find all gifts on that longitude (with deviation sigma)
        sigma = 0.05
        close = find_on_meridian(not_delivered, long, sigma=sigma)
        close = close.sort_values(_PH)

        # Take as many gifts on that linear trip as possible
        while sum(close.Weight) > weight_limit:
            # Drop last FIXME optimazie this case 
            close = close.drop(close.index[len(close.index)-1])

        # Create a trip with selected presents
        t = Trip()
        t.create_trip(close)
        trips.append(t)

        # Mark those gifts as delivered
        not_delivered = not_delivered.drop(close.index)

    total = 0
    for trip in trips:
        total += trip.wrw()

    print 'Total WRW = ', total


    # # Read with additional trip_id column
    # gifts = read_gifts(path)
    #
    # # Use gifts table as a reference
    # # again use gifts as *vec
    # trips = first_level(gifts)
    #
    # # Take copy of trips?
    # second_level(gifts, trips)
