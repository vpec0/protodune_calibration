#!/bin/env python

import sys

import uproot as ur, awkward as ak, numpy as np
from scipy.interpolate import RegularGridInterpolator,CubicSpline



import argparse

import yaml

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('fname',nargs='?')
    parser.add_argument('-e',required=True,help='E-field file',default='')
    parser.add_argument('-c',default='')
    parser.add_argument('-n',type=int,default=100)
    parser.add_argument('--nskip',type=int,default=0) # does not work at the moment
    parser.add_argument('-s',default='')
    parser.add_argument('-o',default='filter_out.root')
    parser.add_argument('--step',type=int,default=1000)
    global args
    args = parser.parse_args()

    if args.fname == None and args.s == '' :
        print('You need to supply path to the input file as a positional argument '+
              'or a text file with list of input files as argument of -s.')
        sys.exit(1)
    if args.fname != None and args.s != '' :
        print('You need to supply either path to the input file as a positional argument'+
              ' or a text file with list of input files as argument of -s. Not both.')
        sys.exit(2)


def run() :
    parse_args()
    global config
    config = argparse.Namespace(**get_config())

    print(f'Got config from {args.c}.')


    # prepare E-field histograms
    global efield_interpolators
    efield_interpolators = GetEfieldInterpolators()
    # Kinetic energy spline
    global kespline
    kespline = GetKESpline()


    # prepare output file
    outf = ur.recreate(args.o)
    PrepareOutputTrees(outf)

    input_files = GetInputFiles()

    nprocessed = 0
    for data in GetData(input_files) :
        print(f'Processed {nprocessed} events')
        if nprocessed >= args.n :
            break
        nprocessed += len(data)
        # zip data to the track level
        data = ak.zip({key:data[key] for key in data.fields},depth_limit=2)
        # trim deepest dimension to match stored number of hits
        data = TrimTrkHits(data)

        # filter data
        dedx_mask = filter_event_for_dedx(data) # masks tracks
        dedx_mask = dedx_mask & filter_plane_for_dedx(data) # masks track planes
        dedx_mask = dedx_mask & filter_ke_pitch(data)

        yz_mask = filter_event_for_yz_corr(data)

        crosser_mask = np.sign(data.trkstartx) != np.sign(data.trkendx)
        # FIXME: don't know why one of the dedx/yz filters need to pass
        x_mask = (dedx_mask | yz_mask) & crosser_mask & filter_hit_x(data)

        # apply common cuts
        mask = filter_trk_angles(data)
        mask = mask & filter_hit_yz_fiducial(data)

        dedx_mask = dedx_mask & mask
        yz_mask = yz_mask & mask
        x_mask = x_mask & mask

        plane=ak.local_index(x_mask,axis=2)
        masks = dict([('def',mask & (dedx_mask|yz_mask|x_mask)),
                      ('dedx',dedx_mask),
                      ('yz',yz_mask),
                      ('x',x_mask)
                      ] +
                     [(f'{l}{i}',m & (plane==i)) for l,m in (('x',x_mask),('yz',yz_mask)) for i in range(3)])

        FillTrees(outf, data, masks)

    # show results - only nonempty events
    #Mask(data,ak.num(data.ntrkhits,axis=1)>0).show(type=True)

    print(f'Closing output file {outf.file_path} with trees and entries:')
    for key,tree in outf.items() :
        print(f'  {key}: {tree.num_entries}')
    outf.close()

def GetData(input_files) :
    # opening input file
    input_files = [l+f':{config.level}/Event' for l in input_files]
    # Get data
    branches = [ f'trk{i}{j}' for i in ['start','end'] for j in 'xyz' ] +\
        [ f'peakT_{i}' for i in ['min','max'] ] +\
        ['trklen', 'adjacent_hits', 'dist_min'] +\
        ['trackthetaxz', 'trackthetayz'] +\
        ['lastwire']+\
        ['ntrkhits','trkdqdx','trkresrange','trkpitch']+\
        [f'trkhit{i}' for i in 'xyz']
    return ur.iterate(input_files, branches, step_size=args.step) #entry_start=args.nskip,entry_stop=args.nskip+args.n)

def FillTrees(outf, d, masks) :
    global tree_branch_names

    # prepare mapping of output tree branch names -> intput tree branch names
    mapping = dict([
        ('dq_dx' , 'trkdqdx'),
        ('resrange','trkresrange')] +\
        [(f'hit_{i}',f'trkhit{i}') for i in 'xyz'] )

    # extract data to be stored which are in the input data
    outd = ak.zip({ newkey:d[oldkey] for newkey,oldkey in mapping.items() })
    # calculate the rest
    outd['efield'] = ak.where(masks['dedx'],get_efield(d), [-999.])
    outd['range_bin'] = ak.values_astype(d.trkresrange/5., 'int64') # res range binned in 5 cm bins
    ke = kespline(ak.ravel(d.trkresrange))
    outd['ke'] = RebuildLike(ak.from_numpy(ke), d.trkresrange)
    outd['hit_plane'] = ak.broadcast_arrays(ak.local_index(d.ntrkhits,axis=2),d.trkresrange)[0]
    outd['dedx_filter'] = ak.values_astype(masks['dedx'],'int64')
    outd['yz_corr_filter'] = ak.values_astype(masks['yz'],'int64')
    outd['x_corr_filter'] = ak.values_astype(masks['x'],'int64')


    # make up data which were not calculated
    for tname,tbns in tree_branch_names.items():
        for key,dtype in tbns.items() :
            if key not in outd.fields :
                outd[key] = ak.zeros_like(outd['ke'],dtype=dtype)
    # reducing dimensionality
    outd = ak.zip({key:ak.ravel(outd[key]) for key in outd.fields})

    # loop over output trees
    for treename in outf.keys() :
        treename=treename.split(';')[0]
        t = outf[treename]
        # get input data
        thistreed = ak.zip({key:outd[key] for key in tree_branch_names[treename].keys()})
        mask = ak.ravel(masks[WhichMask(treename)])
        thistreed = thistreed[mask]
        t.extend(thistreed)

def WhichMask(treename) :
    name_split = treename.split('tree')
    if  name_split == ['',''] :
        return 'def'
    elif name_split == ['dedx_','']:
        return 'dedx'
    elif name_split[0] == 'x_' :
        return f'x{name_split[1]}'
    elif name_split[0] == 'yz_' :
        return f'yz{name_split[1]}'


def filter_trk_angles(d) :
    theta_xz_deg = abs(180./np.pi*d.trackthetaxz)
    theta_yz_deg = abs(180./np.pi*d.trackthetayz)

    plane = ak.local_index(d.ntrkhits, axis=2) # index of the plane
    trk_mask = \
        (((theta_xz_deg>config.plane2_theta_xz_min) &  \
          (theta_xz_deg<config.plane2_theta_xz_max)) | \
         ((theta_yz_deg>config.plane2_theta_yz_min) &  \
          (theta_yz_deg<config.plane2_theta_yz_max)))
    trk_mask = (plane == 2) & trk_mask
    trk_mask = ~trk_mask

    hit_pos = d.trkhitx > 0.

    # FIXME: not sure, why such cut is done on theta_xz
    # mask hits with track in specific angle bands, the mask is to be inverted
    hit_mask_neg =((~hit_pos) & \
                   (((plane == 1) & (theta_xz_deg<config.big_angle)) | \
                    ((plane == 0) & (theta_xz_deg>config.small_angle))) )
    hit_mask_pos = ( hit_pos & \
                     (((plane == 1) & (theta_xz_deg>config.small_angle)) | \
                      ((plane == 0) & (theta_xz_deg<config.big_angle))) )
    hit_mask = trk_mask & ~(hit_mask_neg | hit_mask_pos)
    return hit_mask

def filter_event_for_dedx(d):
    # skip non-crossers
    crosser_mask = np.sign(d.trkstartx) != np.sign(d.trkendx)

    # cuts on peak time, trak length, APA gap
    mask =        (d.peakT_min < config.peakT_min) | (d.peakT_max > config.peakT_max)
    mask = mask | (d.trklen < config.track_len_min) | (d.trklen > config.track_len_max)
    mask = mask | ((d.trkendz > config.track_ediv_min) & (d.trkendz < config.track_ediv_max))
    mask = mask | ((d.trkstartz > config.track_ediv_min) & (d.trkstartz < config.track_ediv_max))
    mask = mask | ((d.trkendz > config.track_ediv2_min) & (d.trkendz < config.track_ediv2_max))
    mask = mask | ((d.trkstartz > config.track_ediv2_min) & (d.trkstartz < config.track_ediv2_max))

    # make sure this is a clean track - no hist close by or made of sparse hits
    mask = crosser_mask & ~(mask | (d.adjacent_hits != 0) | (d.dist_min > 5))

    return mask

def filter_event_for_yz_corr(d) :
    # Check if the start or endpoint are in the FV. Both must be outside to enter sample
    # start is within the fiducial volume
    mask = \
        ((abs(d.trkstartx) < config.track_xmax) & \
         (d.trkstarty > config.track_ymin) & \
         (d.trkstarty < config.track_ymax) & \
         (d.trkstartz > config.track_zmin) & \
         (d.trkstartz < config.track_zmax) )
    # or end is in the FV
    mask = mask | \
        ((abs(d.trkendx) < config.track_xmax) & \
         (d.trkendy > config.track_ymin) & \
         (d.trkendy < config.track_ymax) & \
         (d.trkendz > config.track_zmin) & \
         (d.trkendz < config.track_zmax) )
    # invert the mask
    return ~mask


def filter_plane_for_dedx(d):
    ##Skipping flipped tracks because they have no effect anyways
    mask = RemoveFlippedTracks(d)
    mask = mask & RequireBragg(d) # FIXME: not sure how this works
    return mask

def filter_ke_pitch(d):
    global kespline

    ke = kespline(ak.ravel(d.trkresrange))
    ke = RebuildLike(ak.from_numpy(ke), d.trkresrange)

    mask = (ke>250.) & (ke<450.) & (d.trkpitch>0.5) & (d.trkpitch<0.8)

    return mask

def filter_hit_yz_fiducial(d) :
    mask = (d.trkhity > 0.) & (d.trkhity < config.ymax) & \
        (d.trkhitz > 0.) & (d.trkhitz < config.zmax)
    return mask

def filter_hit_x(d) :
    # Test if enters or exits FV on the positive/negative side of x
    testneg = (d.trkstartx< -config.xmax) | (d.trkendx< -config.xmax)
    testpos = (d.trkstartx>  config.xmax) | (d.trkendx>  config.xmax)
    hit_pos = d.trkhitx > 0.
    nexthitx = Roll(d.trkhitx,1)
    prevhitx = Roll(d.trkhitx,-1)
    # test if neighbouring hits on the same side and this track does not start/end within this side's FV
    mask = testpos & hit_pos & (nexthitx > 0.) & (prevhitx > 0.)
    mask = mask | \
        testneg & (~hit_pos) & (nexthitx < 0.) & (prevhitx < 0.)

    # trim first and last plane hits
    idx = ak.local_index(mask,axis=-1)
    mask=mask & (idx>0) & (idx<ak.num(idx,axis=-1)-1)
    return mask


def PrepareOutputTrees(outf) :
    tree_names = ['tree',
                  [f'yz_tree{i}' for i in range(3)],
                  [f'x_tree{i}' for i in range(3)],
                  f'dedx_tree',
                  ]

    tree_branches = {
        'corrected_dq_dx' : 'float64',
        'dq_dx'           : 'float64',
        'hit_plane'       : 'int64',
        'hit_x'           : 'float64',
        'hit_y'           : 'float64',
        'hit_z'           : 'float64',
        'efield'          : 'float64',
        'resrange'        : 'float64',
        'range_bin'       : 'int64',
        'ke'              : 'float64',
        'dedx_filter'     : 'int64',
        'yz_corr_filter'  : 'int64',
        'x_corr_filter'   : 'int64',
        'Cx'              : 'int64',
        'Cyz'             : 'int64',
    }

    yz_tree_branches = {
        'dq_dx'     : 'float64',
        'hit_plane' : 'int64',
        'hit_x'     : 'float64',
        'hit_y'     : 'float64',
        'hit_z'     : 'float64',
        'efield'    : 'float64',
        'resrange'  : 'float64',
        'range_bin' : 'int64',
        'ke'        : 'float64',
    }

    x_tree_branches = {
      'dq_dx'     : 'float64',
      'hit_plane' : 'int64',
      'hit_x'     : 'float64',
      'hit_y'     : 'float64',
      'hit_z'     : 'float64',
      'efield'    : 'float64',
      'resrange'  : 'float64',
      'range_bin' : 'int64',
      'ke'        : 'float64',
    }

    dedx_tree_branches = {
        'dq_dx'     : 'float64',
        'hit_plane' : 'int64',
        'hit_x'     : 'float64',
        'hit_y'     : 'float64',
        'hit_z'     : 'float64',
        'efield'    : 'float64',
        'resrange'  : 'float64',
        'range_bin' : 'int64',
        'ke'        : 'float64',
    }

    branches = [tree_branches, yz_tree_branches, x_tree_branches, dedx_tree_branches]
    global tree_branch_names
    tree_branch_names = {}
    for tname,tbranches in zip(tree_names, branches) :
        if isinstance(tname, list) :
            for tnamei in tname :
                outf.mktree(tnamei,tbranches)
                tree_branch_names[tnamei] = tbranches
        else :
            outf.mktree(tname,tbranches)
            tree_branch_names[tname] = tbranches


def GetKESpline() :
    spline_range = [0.70437, 1.27937, 2.37894, 4.72636, 7.5788, 22.0917, 30.4441, 48.2235, 76.1461, 123.567, 170.845, 353.438, 441.476]
    spline_ke = [10, 14, 20, 30, 40, 80, 100, 140, 200, 300, 400, 800, 1000]
    return CubicSpline(np.array(spline_range,'d'), np.array(spline_ke,'d'), bc_type='natural')


def GetEFieldHists() :
    efield_file = ur.open(args.e)
    if config.do_true_efield:
        pos_hists = [efield_file[f'True_ElecField_{i}'] for i in ['X', 'Y', 'Z']]
        neg_hists = pos_hists
        #print(pos_hists, neg_hists)
    else:
        pos_hists = [efield_file[f'Reco_ElecField_{i}_Pos'] for i in ['X', 'Y', 'Z']]
        neg_hists = [efield_file[f'Reco_ElecField_{i}_Neg'] for i in ['X', 'Y', 'Z']]

    return pos_hists,neg_hists

def GetInputFiles() :
    if args.s != '' :
        with open(args.s, 'r') as f:
            input_files = [l.strip() for l in f.readlines() if l.strip()[0] != '#']
    else :
        input_files = [args.fname]

    print("Input files:", input_files)
    input_files = [ l if l[:5] != '/pnfs' else l.replace('/pnfs', 'root://fndca1.fnal.gov:1094//pnfs/fnal.gov/usr') for l in input_files ]

    return input_files

def RebuildLike(x,shape) :
    for ax in range(shape.ndim-1,0,-1) :
        x = ak.unflatten(x, ak.ravel(ak.num(shape,axis=ax)))
    return x

def GetEfieldInterpolators():
    pos_hists, neg_hists = GetEFieldHists()
    hists = {'pos':pos_hists,'neg':neg_hists}
    interp = {}
    for key,hxyz in hists.items() :
        tmpi = {}
        for ax,h in zip('xyz',hxyz) :
            vxyz = h.to_numpy()
            x,y,z = tuple([0.5*(i[:-1] + i[1:]) for i in vxyz[1:]])
            tmpi[ax] = RegularGridInterpolator((x,y,z),vxyz[0], bounds_error=False,fill_value=-999)
        interp[key] = tmpi
    return interp

def get_efield(d):
    global efield_interpolators
    interp=efield_interpolators
    E0 = 0.4867
    xyz = np.stack([ak.ravel(d[key]) for key in [f'trkhit{i}' for i in 'xyz']],axis=-1)
    e = ak.zip({ax:ak.where(xyz[...,0]<0., interp['neg'][ax](xyz),interp['pos'][ax](xyz)) for ax in 'xyz'})
    e['x'] = e.x + 1
    e = np.sqrt(e.x**2 + e.y**2 + e.z**2)
    e = e*E0
    return RebuildLike(e, d.trkhitx)

def Roll(data, offsets) :
    idcs = ak.local_index(data, axis=-1)
    # shift indices to the right = subtract offset!
    idcs = (idcs-offsets)%ak.num(data,axis=-1)
    return data[idcs]


def TrimTrkHits(d):
    nhits = ak.from_regular(d.ntrkhits,axis=-1)
    # there's max of 3000 hits
    nhits = ak.where(nhits<3000, nhits, 3000)
    mask = ak.local_index(d.trkhity) < nhits

    return Mask(d, mask)


def RemoveFlippedTracks(d) :
    '''
    Returns True/False for track planes where tracks were not/were flipped. Also, planes with no hits will return False
    This assumes hits ordered in res range, taking first and last hit in each plane
    also assumes hit y to be strictly monotone
    '''
    def getIth(x,i) :
        # need to take into account cases where one plane does not have hits, but other planes do.
        # retruns back array of the same dimension. Last dimension either empty or has the i-th element
        x=ak.mask(x,ak.num(x,axis=-1)>0)
        return ak.singletons(x[...,i],axis=-1)

    hit_y         = getIth(d.trkhity    , 0)
    last_hit_y    = getIth(d.trkhity    ,-1)
    resrange      = getIth(d.trkresrange, 0)
    last_resrange = getIth(d.trkresrange,-1)

    mask = ~((hit_y < last_hit_y) & (resrange > last_resrange))
    return ak.fill_none(ak.firsts(mask,axis=-1), False) # ak.firsts creates bool at plane idx, will have None for empty planes

def RequireBragg(d):
    '''
    Returns True/False for track plane which has/does not have Bragg peak.
    check last 5 cm of the track
    get dqdx at the end of track
    '''
    last_dqdx5 = d.trkdqdx[(d.trkresrange > 0.) & (d.trkresrange < 5.)]
    mask = ak.num(last_dqdx5,axis=-1) >= 5 # require at least 5 hits within the last 5 cm, masks track's plane

    # get dqdx at the start of track
    max_res = ak.max(d.trkresrange,axis=-1) # creates None for empty planes
    first_dqdx5 = d.trkdqdx[(d.trkresrange > 0.) & (d.trkresrange > (max_res - 5.))] # keeps None for empty planes
    first_dqdx5 = ak.fill_none(first_dqdx5,[],axis=2) # recover empty planes
    def median(x) :
        x = ak.sort(x)
        med_idx = ak.singletons(ak.num(x,axis=-1),axis=-1) # empty planes will have 0!
        med_idx = ak.values_astype(med_idx[med_idx>0]/2,int)
        med_idx = med_idx[med_idx > 0] # make planes with 0 or 1 hits empty
        return ak.firsts(x[med_idx],axis=-1) # move the median to the plane dimension, keeps None for empty plane

    med_last = median(last_dqdx5)
    med_first = median(first_dqdx5)

    mask = mask & ak.fill_none((med_first>0.) & (med_last/med_first > 1.4),False,axis=2)
    # mask keeps only False or True at the plane level
    return mask


def Mask(d,mask):
    ''' Dealing with masking an array of records containing arrays.'''
    # First, mask out at common dimensions of the array of records
    tmpmask = mask
    for ax in range(mask.ndim-1, d.ndim-1, -1) :
        tmpmask=ak.any(tmpmask,axis=ax)
    d = d[tmpmask]
    # deal with arrays in the record
    mask = mask[tmpmask]
    for key in d.fields :
        if d[key].ndim == 3 or d[key].ndim <= d.ndim:
            # don't mask out a plane index or branches that had been masked already
            continue
        if d[key].ndim < mask.ndim :
            tmpmask = mask
            for ax in range(mask.ndim-1, d[key].ndim-1, -1) :
                tmpmask=ak.any(tmpmask,axis=ax)
            d[key] = d[key][tmpmask]
        else :
            d[key] = d[key][mask]
    return d

def get_config():
    defaults = {
      'ymax':600.,
      'zmax':695.,
      'level':'michelremoving2',
      'track_xmax':350.,
      'track_ymin':40.,
      'track_ymax':560.,
      'track_zmin':40.,
      'track_zmax':655.,
      'plane2_theta_xz_min':60.,
      'plane2_theta_xz_max':120.,
      'plane2_theta_yz_min':80.,
      'plane2_theta_yz_max':100.,
      'peakT_min':100.,
      'peakT_max':5900.,
      'track_len_max':700.,
      'track_len_min':100.,
      'track_ediv_min':226.,
      'track_ediv_max':236.,
      'track_ediv2_min':456.,
      'track_ediv2_max':472.,
      'do_true_efield':False,
      'big_angle': 140.,
      'small_angle': 40.,
      #'do_combined_efield':false,
    }

    if args.c == '' :
        return defaults

    with open(args.c, 'r') as fin:
        config = yaml.safe_load(fin)

    for k,v in defaults.items():
        if k not in config.keys(): config[k] = v

    return config


def CompareArrays(d1,d2) :
    print('Comparing two arrays')
    print(f'{d1.typestr}, {d2.typestr}')
    if len(d1) != len(d2) :
        print(f'Arrays differ at dimension 0: {len(d1)=},{len(d2)=}')
        return
    for ax in range(1,min(d1.ndim,d2.ndim)) :
        n1 = ak.num(d1,axis=ax)
        n2 = ak.num(d2,axis=ax)
        suma = ak.sum(n1!=n2)
        if (suma):
            print(f'Arrays differ in {suma} elements at depth {ax}:')
            ak.zip((n1[n1!=n2],n2[n1!=n2])).show()
            return

if __name__ == '__main__' :
    run()
