import numpy as np
import pandas as pd

# XXX Longitude is constant for trips from north pole south
# and it's a prominent example of badly named variable
# so we go to more intuitive theta and phi convention
# where theta goes from pole to pole
_TH = 'Longitude'
_PH = 'Latitude'

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
    mass_info = 'Mean value of mass: {0}, with standard deviation: {1}'\
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
