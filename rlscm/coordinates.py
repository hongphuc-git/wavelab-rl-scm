"""Coordinate-first dispersion preprocessing. Units: Hz and rad/m throughout."""
from dataclasses import dataclass,asdict
import numpy as np
@dataclass(frozen=True)
class Grid:
    height:int=256
    width:int=256
    f_min_hz:float=0.0
    f_max_hz:float=2_000_000.0
    k_min_radm:float=0.0
    k_max_radm:float=4000.0
    def __post_init__(self):
        if self.height<2 or self.width<2 or self.f_max_hz<=self.f_min_hz or self.k_max_radm<=self.k_min_radm:
            raise ValueError('Invalid grid dimensions or physical limits')
        if not np.isfinite([self.f_min_hz,self.f_max_hz,self.k_min_radm,self.k_max_radm]).all():
            raise ValueError('Physical limits must be finite')
    def image_extent(self,*,f_scale=1.0,k_scale=1.0):
        """Image outer edges so pixel centers lie on the physical sample nodes."""
        df=(self.f_max_hz-self.f_min_hz)/(self.height-1)
        dk=(self.k_max_radm-self.k_min_radm)/(self.width-1)
        return [(self.k_min_radm-dk/2)*k_scale,(self.k_max_radm+dk/2)*k_scale,
                (self.f_min_hz-df/2)*f_scale,(self.f_max_hz+df/2)*f_scale]
    def metadata(self):return {**asdict(self),'row_axis':'frequency increasing','column_axis':'angular wavenumber increasing','units':['Hz','rad/m'],'rounding':'nearest node, numpy rint; out-of-window excluded','occupancy':'binary; coincident points merged'}

def pixels_to_physical(rows,cols,source,*,index_base=0,frequency_increases_with_row=True):
    """Source image indices -> physical f,k; caller must exclude sparse padding."""
    rows=np.asarray(rows,dtype=float);cols=np.asarray(cols,dtype=float)
    if rows.shape!=cols.shape:raise ValueError('Row and column shapes differ')
    if index_base not in (0,1):raise ValueError('index_base must be 0 or 1')
    r=rows-index_base;c=cols-index_base
    if not (np.isfinite(r).all() and np.isfinite(c).all() and np.equal(r,np.rint(r)).all() and np.equal(c,np.rint(c)).all()):raise ValueError('Pixel indices must be finite integers')
    if np.any((r<0)|(r>=source.height)|(c<0)|(c>=source.width)):raise ValueError('Pixel outside source grid')
    rf=r/(source.height-1)
    if not frequency_increases_with_row:rf=1-rf
    return source.f_min_hz+rf*(source.f_max_hz-source.f_min_hz),source.k_min_radm+c/(source.width-1)*(source.k_max_radm-source.k_min_radm)

def rasterize_physical(f_hz,k_radm,grid=Grid()):
    """Physical points -> binary image; reject outside window before rounding."""
    f=np.asarray(f_hz,dtype=float);k=np.asarray(k_radm,dtype=float)
    if f.shape!=k.shape:raise ValueError('Frequency and wavenumber shapes differ')
    keep=np.isfinite(f)&np.isfinite(k)&(f>=grid.f_min_hz)&(f<=grid.f_max_hz)&(k>=grid.k_min_radm)&(k<=grid.k_max_radm)
    r=np.rint((grid.height-1)*(f[keep]-grid.f_min_hz)/(grid.f_max_hz-grid.f_min_hz)).astype(int)
    c=np.rint((grid.width-1)*(k[keep]-grid.k_min_radm)/(grid.k_max_radm-grid.k_min_radm)).astype(int)
    image=np.zeros((grid.height,grid.width),dtype=np.int8);image[r,c]=1
    return image,keep
