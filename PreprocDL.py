import os
from os.path import basename
import sys, argparse
import numpy as np
import h5py
import cv2
import PIL
import matplotlib.pyplot as plt
import constants as cc
from io import BytesIO
from scipy import stats
from pylab import *
import scipy
from scipy import ndimage
import scipy.integrate as si
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.ticker import *
from scipy.constants import c
from matplotlib import cm

class HDF_Data: 
    '''Loading Class hdf5 format data file'''
    dtype = [('array','f8',(200,200)),('xH','f8'),('Box','i8'),('z','f8'),('index','i8')]
    def __init__(self, file):
        self.__file = None
        self.__IO = None
        self.open(file)

    def open(self, path = None):
        '''create new hdf5 format data. if exist, open hdf5 '''
        if not self.__file is None:
            return
        if path is None:
            self.__file = h5py.File(self.system.path+self.system.name+'.hdf5','r')
        else:
            self.__file = h5py.File(path, 'r')

    def close(self):
        #del self.__file
        if not self.__file is None:
            self.__file.close()
            self.__file = None

    def set_file(self, h5obj):
        self.__file = h5obj

    def __del__(self):
        del self.__file

    def isfile(self, path = None):
        if path is None:
            return os.path.isfile(self.system.path+self.system.name+'.hdf5')
        else:
            return os.path.isfile(path)
        
    def npy_merger(self, ndarray):
        if not isinstance(ndarray,np.ndarray):
            ndarray = np.load(ndarray)
        dtype = [('array','f8',(200,200)),('xH','f8'),('Box','i8'),('z','f8'),('index','i8')]
        temp = np.empty(ndarray.shape, dtype = dtype)
        for i,dline in enumerate(ndarray):
            temp[i]['xH'] = dline['xH']
            temp[i]['Box'] = dline['Box']
            z = float(dline['z'][2:])
            temp[i]['z'] = z
            temp[i]['index'] = dline['index']
            if self.is_exist('{}/{}/{}'.format(dline['Box'],z,dline['index'])):
                temp[i]['array'] = self.file['{}/{}/{}'.format(dline['Box'],z,dline['index'])]
            else:
                temp[i]['array'] = self.file['{}/{}/cubic'.format(dline['Box'],z)][:,:,dline['index']]
        return DeltaTarray(temp)
    
    def xH_merger(self, xH):
        '''xH must be cubic hdf5 data.'''
        if not isinstance(xH, self.__class__):
            xH = HDF_Data(xH)
        length = 0
        for box in self.file:
            for z in self.file[box]:
                if self.is_exist('cubic'):
                    length += 200
                else:
                    length += len(self.file[box][z])
                    
        dtype = [('array','f8',(200,200)),('xH','f8'),('Box','i8'),('z','f8'),('index','i8')]
        temp = np.empty(length, dtype = dtype)            
        index = 0
        for box in self.file.keys():
            for z in self.file[box].keys():
                xHmean = xH.file[box][z]['cubic'][:].mean(axis=(0,1))
                if self.is_exist('cubic'):
                    for i in range(200):
                        temp[index]['array'] = self.file[box][z]['cubic'][:,:,i]
                        temp[index]['index'] = i
                        temp[index]['xH'] = xHmean[i]
                        temp[index]['Box'] = int(box)
                        temp[index]['z'] = float(z)
                        index+=1
                else:
                    for ind in self.file[box][z].keys():
                        i = int(ind)
                        temp[index]['array'] = self.file[box][z][ind]
                        temp[index]['index'] = i
                        temp[index]['xH'] = xHmean[i]
                        temp[index]['Box'] = int(box)
                        temp[index]['z'] = float(z)
                        index += 1
        return DeltaTarray(temp)
    
    def is_exist(self, target_path):
        '''return True when Data exist on given path.'''
        ch  = not (self.file.get(target_path, None) is None)
        return ch

    def ls(self, path = None, tapping = '', show_all = False):
        if path is None:
            target = self.__file
            print(self.file.filename)
        else:
            target = self[path]
            print(tapping+target.name+" : dir")
        tapping += '\t'
        for name, item in list(target.items()):
            if isinstance(item, h5py.Group):
                if show_all:
                    self.ls(item.name, tapping, show_all)
                else:
                    print(tapping+item.name+" : dir")
            else:
                print(tapping+name)

    @property
    def file(self):
        return
    @file.getter
    def file(self):
        if self.__file is None:
            self.open()
        return self.__file

    @property
    def attrs(self):  return self.open()
    @attrs.getter
    def attrs(self):
        return self.file.attrs
    def require_group(self, *arg,**kwarg): return self.file.require_group(*arg,**kwarg)
    def get(self, *arg,**kwarg): return self.file.get(*arg,**kwarg)
    def flush(self):          return self.file.flush()
    def items(self):          return self.file.items()
    def keys(self):           return self.file.keys()
    def values(self):         return self.file.values()
    def __getitem__(self, key):         return self.file[key]
    def __setitem__(self, key, value):
        if self.is_exist(key):
            self.file[key] = value
        else:
            self.files.attrs[key] = value

class HyperParameter:
    cosmo = {'omega_M_0':0.31, 'omega_lambda_0':0.69, 'omega_k_0':0.0, 'h':0.68}
    ## if you revise the cosmo parameters to the other values 
    ## You must change these parameters on "def bw_to_lbox" defaults arguments
    def __init__(self, dtype = np.float32):
        self.raw_data = []
        self.Box = []
        self.redshift = []
        self.filename = []
        self.filter = []
        self.index = []
        self.target = []
        self.DIM = []
        self.dtype  = dtype
        self.Lboxx_xy = []
        self.Lboxx_z = []
        self.codis = []
        self.tar_redshift = []
        self.tar_codis = []
        
    def append_folder(self, folder_path):
        abspath = os.path.abspath(folder_path)+'/'
        bstart = 'Box'
        bend = '_Dim'
        boxsize = lambda x: x[x.find(bstart)+len(bstart):x.find(bend)]
        dstart = 'HIIDIM'
        dend = '_zstart'
        DIM = lambda x: x[x.find(dstart)+len(dstart):x.find(dend)]
        pathlist = os.listdir(folder_path)
        zstart = '_z'
        zend = '_nf'
        redshift = lambda x: x[x.find(zstart)+len(zstart):x.find(zend)]
        for path in pathlist:
            if len(redshift(path))<8:
                if path in self.filename: continue
                self.raw_data.append(abspath+path)
                self.Box.append(int(boxsize(folder_path)))
                self.redshift.append(float(redshift(path)))
                self.filename.append(path)
                self.DIM.append(int(DIM(folder_path)))
    
    def set_parameter(self, bandwidth, beamsize):
        def _comoving_integrand(z, omega_M_0, omega_lambda_0, omega_k_0, h):
            e_z = (omega_M_0 * (1+z)**3. +  omega_k_0 * (1+z)**2. +   omega_lambda_0)**0.5     
            H_0 = h * cc.H100_s 
            H_z =  H_0 * e_z            
            return cc.c_light_Mpc_s / (H_z)  
        comoving = lambda z, z0, omega_M_0, omega_lambda_0, omega_k_0, h: si.quad(_comoving_integrand, z0, z, limit=1000,args=(omega_M_0, omega_lambda_0, omega_k_0, h))
        
        # Must take the theta as unit of arcmin
        def Lbox(d_co, theta):
            Lbox = d_co * (theta/3437.75)
            return Lbox
        def bw_to_lbox(bw,z,h=0.68,omega_m=0.31,omega_lamb=0.69):
            hubble_h = 100.*h
            L_box = (bw * c*((1.+z)**2))/(1000*(1420*hubble_h*np.sqrt(omega_m*((1+z)**3)+omega_lamb)))
            return L_box
        def fwhm_sigma(fwhm):
            sigma = fwhm/(2*np.sqrt(2*np.log(2)))
            return sigma
        
        co_dist = []
        tar_codist = []
        for i in self.redshift:
            co_dist.append(comoving(i, 0, **self.cosmo)[0])
        
        for i in self.tar_redshift:
            tar_codist.append(comoving(i, 0, **self.cosmo)[0])
            
        self.tar_codis = tar_codist
        self.codis = co_dist # TODO
    
        assert len(co_dist) == len(self.Box), "Nope!"
        filterxy = []
        filterz= []
        for box, codi, z in zip(self.Box, co_dist, self.redshift):
            filterxy.append(round(fwhm_sigma(Lbox(codi, beamsize)/(box/200)), 2))
            self.Lboxx_xy.append(Lbox(codi, beamsize)) # TODO
            filterz.append(round(fwhm_sigma(bw_to_lbox(bandwidth, z)/(box/200)), 2))
            self.Lboxx_z.append(bw_to_lbox(bandwidth, z))
        self.filter = []
        for xy, z in zip(filterxy, filterz):
            self.filter.append((xy, xy, z))
        #####################################
    
    def get_angle(self):
        ang_dia_dist = []
        angle = []
        func_ang_dia_dist = lambda r, z: r/(1+z)
        func_angle = lambda object_size, ang_dia_dist: object_size/ang_dia_dist
        
        for co_dist, redshift in zip(self.tar_codis, self.tar_redshift):
            ang_dia_dist.append(func_ang_dia_dist(co_dist, redshift))
        
        for dist in ang_dia_dist:
            angle.append(func_angle(50, dist))
        
        return angle, ang_dia_dist
    
    def add_tar_redshift(self, list_thing):
        '''
            input type just targeting redshift float list
        '''
        for argu in list_thing:
            self.tar_redshift.append(argu)
    
    def add_target(self, **kwargs): ## Indicate the redshifts that I make for each Box lengths
        '''input type box{boxsize}:[redshifts]'''
        for arg in kwargs:
            if arg[:3].lower() == 'box':
                box = int(arg[3:])
                for i, b, z in zip(range(len(self.Box)), self.Box, self.redshift):
                    if box == b and z in kwargs[arg]:
                        self.target.append(i)
    
    def indexing(self, index): ## Indicate the Index in redshifts that I make for
        if not isinstance(index, list): raise TypeError("Please enter the index as list type")
        
        self.index = np.array(index)
        if (self.index<0).all() : raise ValueError("Index must be larger than 0")
                        
    def clear_target(self):
        self.target  = []
        
    def see_target(self):
        for i in self.target:
            print("filename : {} \n\tBox size : {}\n\tRedshift : {}".format(self.filename[i],self.Box[i],self.redshift[i]))
            if self.filter:
                print("\tfilter(x,y,z) : {}".format(self.filter[i]))
    
    def make_array(self, file, get_raw = False, verbose = False):
        saver = h5py.File(file,'a')
        if len(self.target)==0:
            self.target = range(len(self.Box))
            
        for i in self.target:
            print('Processing input file:')
            print('\t{}'.format(self.filename[i]))
            f = open(self.raw_data[i], 'rb')
            data = f.read()
            f.close()
            data = np.fromstring(data, self.dtype)
            if sys.byteorder == 'big':
                data = _data.byteswap()
            data.shape = (self.DIM[i], self.DIM[i], self.DIM[i])
            data = data.reshape((self.DIM[i], self.DIM[i], self.DIM[i]), order='F')
            
            if get_raw:
                box = saver.require_group(str(self.Box[i]))
                redshift = box.require_group(str(self.redshift[i]))
                for index in self.index:
                    redshift.create_dataset(str(index),data = data[:,:,index])
                if len(self.index) == 0 :
                    redshift.create_dataset('cubic', data = data)
                continue
            
            if self.filter[i][0] >= 0:
                x_sigma = self.filter[i][0]
            if verbose:
                print("\t\tSmoothing along the x (horizontal) axis with a Gaussian filter of width ="+str(x_sigma))
            data = scipy.ndimage.filters.gaussian_filter1d(data, sigma=x_sigma, axis=1, mode='wrap')
            if self.filter[i][1] >= 0:
                y_sigma = self.filter[i][1]
            if verbose:
                print("\t\tSmoothing along the y (vertical) axis with a Gaussian filter of width ="+str(y_sigma))
            data = scipy.ndimage.filters.gaussian_filter1d(data, sigma=y_sigma, axis=0, mode='wrap')
            if self.filter[i][2] >= 0:
                z_sigma = self.filter[i][2]
            if verbose:
                print("\t\tSmoothing along the LOS (z) axis with a Gaussian filter of width ="+str(z_sigma))
            data = scipy.ndimage.filters.gaussian_filter1d(data, sigma=z_sigma, axis=2, mode='wrap')

            box = saver.require_group(str(self.Box[i]))
            redshift = box.require_group(str(self.redshift[i]))
            for index in self.index:
                redshift.create_dataset(str(index),data = data[:,:,index])
            if len(self.index) == 0 :
                redshift.create_dataset('cubic', data = data)
        
        saver.close()
        print("\n\t\t Making array all Done")
        return HDF_Data(file)
        '''ANS = input("\n\t Return HDF_Data instance?? (Y/N) : ")
        if ANS == 'Y':
            instance_name = input("\n\t Enter the instance name of HDF_Data : ")
            globals()[instance_name] = HDF_Data(file) 
        else:
            return "Ok, Only save the hdf5 data"'''
        
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# 2019. 08. 29 revised
# Make all delta_T array having zero mean distribution
class DeltaTarray: 
    '''
        이 클라스는 HyperParameter 로부터 추출해낸 delta_Tb array를 가지고 가공하는 클라스다.
    '''
    cmap = LinearSegmentedColormap.from_list('mycmap', ['yellow','red','black','green','blue'])
    norm = MidpointNormalize(midpoint=0)
    def __init__(self, ndarray):
        '''
            Basically must have dtype([('array', '<f8', (200, 200)), ('xH', '<f8'), ('index', '<i8')])
        '''
        if not isinstance(ndarray, np.ndarray): 
            raise TypeError (type(ndarray), "is not allowed, please Try again for type <numpy.ndarray>")
        if not ndarray.dtype.names:
            raise TypeError (type(ndarray.dtype), "is not allowed, Basically must have dtype([('array', '<f8', (200, 200)), ('xH', '<f8'), ('index', '<i8')])")
        if any(np.logical_not([dtype in ndarray.dtype.names for dtype in ('array','xH','index')])):
            raise TypeError (ndarray.dtype, "is not allowed, Basically must have dtype([('array', '<f8', (200, 200)), ('xH', '<f8'), ('index', '<i8')])")
        self.origin = ndarray
    
    def dtype(self):
        return self.origin.dtype
    
    def DeltaT_merger(self, in_DeltaTarray):
        '''
            DeltaTarray instance merge.
        '''
        if not isinstance(in_DeltaTarray, self.__class__):
            in_DeltaTarray = DeltaTarray(in_DeltaTarray)
        
        if self.origin.dtype != in_DeltaTarray.origin.dtype:
            print("SELF data dtype : ", self.origin.dtype)
            print("Input data dtpye : ", in_DeltaTarray.origin.dtype)
            print("were not matched")
            raise TypeError ("SELF data dtype : {} \nINPUT data dtype : {} \n were not matched.".format(self.origin.dtype, in_DeltaTarray.origin.dtype))
        
        Merged_array = np.append(self.origin, in_DeltaTarray.origin, axis=0)
        return DeltaTarray(Merged_array)
    
    
    def get_numpy(self):
        return self.origin
    
    def save(self, filename):
        '''
            Save the data as numpy array(.npy)
        '''
        np.save("{}".format(str(filename)),self.origin)

    def load(self, filename):
        '''
            Load the numpy array as DeltaTarray instance.
        '''
        file = np.load("{}".format(str(filename)))
        return DeltaTarray(file)
        
    def convert_to_stdDist(self):
        '''
            All the delta Tb temperature array make them having a zero mean distribution.
        '''
        temp = self.origin.copy()
        delT_array = self.origin['array'].reshape(self.origin.shape[0], -1)
        delT_mean = self.origin['array'].mean(axis=(1,2))
        result = delT_array.T - delT_mean
        temp['array'] = result.T.reshape(self.origin['array'].shape)
        return DeltaTarray(temp)
    
    def __getitem__(self, key): ## == self[key] -> self.origin[key]
        return self.origin[key]
    
    def plot(self, index, dpi=72, vmin=-210, vmax=30, saveimage=False, *arg,**kwarg):
        '''
            Simply the temperature 21cm signal value plot.
        '''
        target = self.origin[index]
        print("Box : {}, index : {}, \n\t   xH : {},\n\t   redshift(z) : {}".format(target['Box'],target['index'], target['xH'], target['z'],2))
        plt.clf()
        plt.figure(dpi=dpi)
        delT_map = plt.imshow(self.origin['array'][index], cmap=DeltaTarray.cmap, norm=DeltaTarray.norm, *arg, **kwarg)
        delT_map.set_clim(vmin=vmin, vmax=vmax)
        delT_map.axes.get_xaxis().set_visible(False)
        delT_map.axes.get_yaxis().set_visible(False)
        plt.axis("off")
        plt.show()
        if saveimage:
            IMG_SAVE_FOLDER_PATH = './Img_repo/'
            os.mkdir(IMG_SAVE_FOLDER_PATH, exist_ok=True)
            plt.savefig('{}{}.png'.format(IMG_SAVE_FOLDER_PATH, index), bbox_inches='tight', pad_inches=0)
        
    def Value_to_Img(self, dpi=72, Grayscale=False, IMG_SIZE=200, *arg, **kwarg):
        '''
            Make the 21cm Signal array into Image array(RGB or GRAYSCALE)
        '''
        if Grayscale:
            dtype = [('Img','i8',(200,200)),('xH','f8'),('Box','i8'),('z','f8'),('index','i8')]
        else:
            dtype = [('Img','i8',(200,200,3)),('xH','f8'),('Box','i8'),('z','f8'),('index','i8')]
        
        temp = np.empty(self.origin.shape, dtype=dtype)
        temp['xH'] = self.origin['xH']
        temp['Box'] = self.origin['Box']
        temp['z'] = self.origin['z']
        temp['index'] = self.origin['index']
        print("done")
        print("done")
        
        if Grayscale:
            for idx, dline in enumerate(self.origin):
                plt.figure(dpi=dpi)
                plt.imshow(self.origin[idx]['array'], cmap=plt.get_cmap('gray'), norm=DeltaTarray.norm)
                plt.clim(vmin=-210, vmax=30)
                plt.subplots_adjust(0,0,1,1)
                plt.axis('off')
                buffer_ = BytesIO()
                plt.savefig(buffer_, format='png', bbox_inches='tight', pad_inches=0)
                image = PIL.Image.open(buffer_)
                image = image.convert('L')
                img_array = np.asarray(image)
                buffer_.close()
                GRAY_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                temp[idx]['Img'] = GRAY_array
                plt.clf()
                plt.cla()
                plt.close()
                if idx % 1000 == 0:
                    print(" ", idx)
        
        else:
            for idx, dline in enumerate(self.origin):
                plt.clf()
                plt.cla()
                plt.figure(dpi=dpi)
                plt.imshow(self.origin[idx]['array'], cmap=DeltaTarray.cmap, norm=DeltaTarray.norm)
                plt.clim(vmin=-210, vmax=30)
                plt.subplots_adjust(0,0,1,1)
                plt.axis('off')
                buffer_ = BytesIO()
                plt.savefig(buffer_, format='png', bbox_inches='tight', pad_inches=0)
                image = PIL.Image.open(buffer_)
                image = image.convert('RGB')
                img_array = np.asarray(image)
                RGB_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                temp[idx]['Img'] = RGB_array
                buffer_.close()
                plt.clf()
                plt.cla()
                plt.close()
                if idx % 1000 == 0:
                    print(" ", idx)
        
        return temp

    def MakesomeNoise_White(self, rms, mean=0, SIZE=200):
        '''
            White Gaussian Noise, if mean=0 rms is equal to std(sigma).
        '''
        temp = self.origin.copy()
        White_Noise = stats.distributions.norm.rvs(mean, rms, size=self.origin['array'].shape)
        Noised_array = self.origin['array'] + White_Noise
        temp['array'] = Noised_array
        return DeltaTarray(temp)
    
    def MakesomeNoise_Uniform(self, rms, SIZE=200):
        '''
            sigma^2 = (2N)^2 / 12 , if mean=0 rms is equal to std(sigma).
        '''
        N = round(np.sqrt(3*(rms**2)))
        temp = self.origin.copy()
        Uniform_Noise = np.random.uniform(-N, N, size=self.origin['array'].shape)
        Noised_array = self.origin['array'] + Uniform_Noise
        temp['array'] = Noised_array
        return DeltaTarray(temp)