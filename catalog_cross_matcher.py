import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, Column, unique, vstack, TableColumns
from astropy import table

def catalog_cross_matcher(main_cat, other_cat_table, other_cat_name, ra_column_name_main, dec_column_name_main, ra_column_name_other, dec_column_name_other, add_cols, max_distance=0.5):
    
    if main_cat[ra_column_name_main].unit == 'deg' and main_cat[dec_column_name_main].unit == 'deg':
        main = SkyCoord(ra = main_cat[ra_column_name_main], dec = main_cat[dec_column_name_main])
    else:
        main = SkyCoord(ra = main_cat[ra_column_name_main]*u.deg, dec = main_cat[dec_column_name_main]*u.deg)
    
    if other_cat_table[ra_column_name_other].unit == 'deg' and other_cat_table[dec_column_name_other].unit == 'deg':
        other = SkyCoord(ra = other_cat_table[ra_column_name_other], dec = other_cat_table[dec_column_name_other])
    else:
        other = SkyCoord(ra = other_cat_table[ra_column_name_other]*u.deg, dec = other_cat_table[dec_column_name_other]*u.deg)

    index, d2d, d3d = main.match_to_catalog_sky(other)
    
    main_cat.add_column(col = np.zeros(len(main_cat), dtype = 'int'), name = f"{other_cat_name}_match_number")
    main_cat.add_column(col = np.zeros(len(main_cat), dtype = 'float'), name = f"{other_cat_name}_match_separation")
    
    for i in range(len(main_cat)):
        main_cat[f"{other_cat_name}_match_number"][i] = index[i]
        main_cat[f"{other_cat_name}_match_separation"][i] = (d2d[i].to(u.arcsec)) / u.arcsec
    
    indices = main_cat[f'{other_cat_name}_match_number']
    
    if add_cols != None:
        for i in range(len(add_cols)):
            main_cat.add_column(col = other_cat_table[add_cols[i]][indices], name = add_cols[i])
    else:
        pass
    
    match_numbers = np.unique(main_cat[f'{other_cat_name}_match_number'])

    main_cat.add_column(col=np.ones(len(main_cat), dtype='bool'), name=f'{other_cat_name}_matched')
    
    for idnumber in match_numbers:
        selection = main_cat[f'{other_cat_name}_match_number'] == idnumber
        separations = main_cat[f'{other_cat_name}_match_separation'][selection]
        closest = np.where(selection)[0][np.argmin(separations)]
        selection[closest] = False
        for col in add_cols:
            main_cat[col][selection] = np.nan
        main_cat[f'{other_cat_name}_matched'][selection] = False
        
    for i in range(len(main_cat)):
        if main_cat[i][f'{other_cat_name}_matched'] == True and main_cat[i][f'{other_cat_name}_match_separation'] > max_distance:
            main_cat[i][f'{other_cat_name}_matched'] = False
            for col in add_cols:
                main_cat[i][col] = np.nan
        
    return main_cat