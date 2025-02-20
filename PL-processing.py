
import datetime
import time
from collections import namedtuple
import arcpy
import arcpy.conversion
import arcpy.ia
import tomli as toml
import os
import arcpy
import numpy as np
import traceback

is_test = False


class Timer:
    """Simple timer for determining execution time"""
    def __init__(self) -> None:
        self.start = time.perf_counter()
    def time_passed(self) -> int:
        """Return time passed since start, in seconds"""
        return time.perf_counter() - self.start
    def get_time(self) -> tuple:
        """Returns durations in hours, minutes, seconds as a named tuple"""
        duration = self.time_passed()
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        PassedTime = namedtuple('PassedTime', 'hours minutes seconds')
        return PassedTime(hours, minutes, seconds)
    def get_duration(self) -> str:
        """Returns a string of duration in hours, minutes, seconds"""
        t = self.get_time()
        return '%d h %d min %d sec' % (t.hours, t.minutes, t.seconds)
    def get_hhmmss(self) -> str:
        """Returns a string of duration in hh:mm:ss format"""
        t = self.get_time()
        return '[' + ':'.join(f'{int(value):02d}' for value in [t.hours, t.minutes, t.seconds]) + ']'
    def reset(self) -> None:
        """Reset timer"""
        self.start = time.perf_counter()

#checked
def exception_traceback(e: Exception):
    """
    Format exception traceback and print
    """
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    aprint('--------------')
    aprint(''.join(tb))

#checked
def aprint(s: str):
    """
    Short for arcpy.AddMessage()
    """
    arcpy.AddMessage(s)
#checked
def delete_fc(fc: str):
    """
    Short for arcpy.Delete_management()
    """
    try:
        if not is_test:
            if arcpy.Exists(fc):
                arcpy.Delete_management(fc)
    except Exception as e:
        arcpy.AddWarning(f'WARNING. Can not delete temporary file {fc}.')
#checked            
def delete_all_fc():
    """
    Delete all temporary datasets
    """
    try:
        for fc in arcpy.ListFeatureClasses() + arcpy.ListRasters():
            delete_fc(fc)
    except Exception as e:
        arcpy.AddWarning(f'WARNING. Can not delete temporary files.')
#checked
def nan_to_none(input):
    """
    Arguments:
        input (dict | list): input iterable

    Returns:
        dict or list with all nan values converted to None
    """
    try:
        if isinstance(input, dict):
            return { key: (nan_to_none(item) if isinstance(item, (dict, list)) else (item if (not isinstance(item, float) or not np.isnan(item)) else None)) for key, item in input.items() }
        elif isinstance(input, list):
            return [ (nan_to_none(item) if isinstance(item, (dict, list)) else (item if (not isinstance(item, float) or not np.isnan(item)) else None)) for item in input ]
        else:
            raise Exception(f'Cannot process given data type: {type(input)}')
    except Exception as e:
        arcpy.AddError(e)
        exit()
#checked
def field_exists(dataset, field_name):
    """
    Checks if field exists in the dataset
    Arguments:
        dataset: dataset path
        field_name: name of the attribute

    Returns:
        exists: True or False
    """
    fields = arcpy.ListFields(dataset)
    exists = False

    for field in fields:
        if field.name == field_name:
            exists = True
            
    return exists
#checked
def create_copy(layer: str) -> str:
    """
    Creates a copy of the layer

    Arguments:
        layer (str): feature class name
    
    Returns:
        layer_copy (str): feature class copy name
    """
    aprint('Copying...')
    layer_copy = layer + '_copy'
    arcpy.CopyFeatures_management(layer, layer_copy)
    return layer_copy

#checked
def create_buffer(layer: str, instructions: dict[str, any]) -> str:
    """
    Creates a buffer area around each object in the given layer

    Arguments:
        layer (str): feature class name
        instructions (dict): properties of operation to be performed
    
    Returns:
        layer_buffered (str): buffered polygons feature class name
    """
    aprint('Creating buffer...')
    # buffer calculation rules
    def get_buffer_distance(attr_val):
        dist = None
        if attr_val == None:
            dist = instructions['default']
        else:
            for i in range(len(instructions['threshold'])):
                threshold = instructions['threshold'][i]
                if dist == None:
                    dist = instructions['value'][i]
                if attr_val > threshold:
                    dist = instructions['value'][i+1]
        return dist
    # get buffer distances and process buffers
    if isinstance(instructions['value'], list):
        arcpy.AddField_management(layer, 'buffer_distance', ('TEXT' if isinstance(instructions['value'][0], str) else 'SHORT'))
        # if the buffer distance depends on attribute value
        if not field_exists(layer, instructions['attribute']):
            check_field = instructions['attribute']
            arcpy.AddError(f'ERROR. Mandatory field "{check_field}" does not exist in the input dataset.')
            delete_all_fc()
            exit()
        with arcpy.da.UpdateCursor(in_table=layer, field_names=['buffer_distance', instructions['attribute']]) as cursor:
            for row in cursor:
                row[0] = get_buffer_distance(row[1])   # conditional buffer distance
                cursor.updateRow(row)
    else:
        arcpy.AddField_management(layer, 'buffer_distance', ('TEXT' if isinstance(instructions['value'], str) else 'SHORT'))
        # single value buffer distance
        with arcpy.da.UpdateCursor(in_table=layer, field_names=['buffer_distance']) as cursor:
            for row in cursor:
                row[0] = instructions['value']     # single value buffer distance (int)
                cursor.updateRow(row)
    # create buffer and overwrite layer
    layer_buffered = layer+'_buffer'
    arcpy.Buffer_analysis(in_features=layer, out_feature_class=layer_buffered, buffer_distance_or_field='buffer_distance')
    return layer_buffered

#checked
def create_rings(layer: str, instructions: dict[str, any]) -> str:
    """
    Creates buffer rings around each object in the given layer

    Arguments:
        layer (str): feature class name
        instructions (dict): properties of operation to be performed

    Returns:
        layer_rings (str): buffer rings polygon feature class name
    """
    aprint('Creating buffer rings...')
    layer_rings = layer+'_buffer_rings'
    # create a new field for the polygon value (without multiplier)
    temp_field = 'temp_field'
    arcpy.AddField_management(in_table=layer, field_name=temp_field, field_type='FLOAT')
    attr = 'HA_value'
    temp_rings = 'temp_rings'
    # if there are several attributes to consider (e.g. data from several years)
    aprint('\tCalculating polygon values...')
    if isinstance(instructions['value'], list):
        # select the fields which fit the pattern
        fields = [field.name for field in arcpy.ListFields(layer, f'{instructions["value"][1]}*')]
        default_value = None
        # define the default value
        if instructions['default'] in ['mean', 'median']:
            arr = np.array([row for row in arcpy.da.SearchCursor(layer, fields)], dtype=np.float32)
            if instructions['default'] == 'mean':
                default_value = np.nanmean(arr)
            elif instructions['default'] == 'median':
                default_value = np.nanmedian(arr)
            else:
                arcpy.AddWarning(f'WARNING. Unknown default value: {instructions["default"]} is given in the configuration file. No processing can be performed.')
                return None
            if np.isnan(default_value):
                if 'nodata' in instructions:
                    default_value = instructions['nodata']
                else:
                    arcpy.AddWarning(f'WARNING. All values are empty for the dataset. No processing can be performed.')
                    return None
        else:
            default_value = instructions['default']
        # set the value for each row
        with arcpy.da.UpdateCursor(layer, fields+[temp_field]) as cursor:
            for row in cursor:
                # if none of the row's attributes have a value
                arr = np.array([row[i] for i in range(len(fields))], dtype=np.float32)
                if np.sum(~np.isnan(arr)) == 0:
                    row[len(fields)] = float(default_value)
                # otherwise, concatenate the values by defined function
                elif instructions['value'][0] == 'average':
                    row[len(fields)] = float(np.nanmean(arr))
                else:
                    arcpy.AddWarning(f'WARNING. Cannot process default value: {instructions["value"][0]}. No processing can be performed.')
                    return None
                cursor.updateRow(row)
    # single defined value (not attribute)
    else:
        with arcpy.da.UpdateCursor(layer, [temp_field]) as cursor:
            for row in cursor:
                row[0] = instructions['value']
                cursor.updateRow(row)
    aprint('\tCreating individual rings...')
    # create buffer for the innermost ring
    arcpy.Buffer_analysis(layer, temp_rings+str(0), instructions['range'][0])
    # create a new field for the HA value, and set the value for each row
    arcpy.AddField_management(in_table=temp_rings+str(0), field_name=attr, field_type='FLOAT')
    with arcpy.da.UpdateCursor(in_table=temp_rings+str(0), field_names=[attr, temp_field]) as cursor:
        for row in cursor:
            row[0] = instructions['multiplier'][0] * row[1]
            cursor.updateRow(row)
    rings = [temp_rings+str(0)]
    # for every additional ring
    for i in range(1, len(instructions['range'])):
        # create a temporary buffer, from point centre up to inner boundary of ring
        temp_temp = 'temp_temp' + str(i)
        arcpy.Buffer_analysis(layer, temp_temp, instructions['range'][i-1])
        # create the ring as a buffer from the temporary buffer, and set the buffer to be 'OUTSIDE_ONLY'
        arcpy.Buffer_analysis(temp_temp, temp_rings+str(i), instructions['range'][i]-instructions['range'][i-1], 'OUTSIDE_ONLY')
        rings.append(temp_rings+str(i))
        delete_fc(temp_temp)
        # add a field for HA value, and set the value for each row
        arcpy.AddField_management(in_table=temp_rings+str(i), field_name=attr, field_type='FLOAT')
        with arcpy.da.UpdateCursor(in_table=temp_rings+str(i), field_names=[attr, temp_field]) as cursor:
            for row in cursor:
                row[0] = instructions['multiplier'][i] * row[1]
                cursor.updateRow(row)
    aprint('\tMerging ring layers to one...')
    arcpy.Merge_management(rings, layer_rings)
    for ring in rings:
        delete_fc(ring)
    return layer_rings

#checked
def replace_layer_values(layer: str, field: str, old_values: list[float], new_value: float):
    """
    Sets all instances of old values in field to new value
    """
    old_values = np.array(old_values)
    if not field_exists(layer, field):
        arcpy.AddError(f'ERROR. Mandatory field "{field}" does not exist in the input dataset.')
        delete_all_fc()
        exit()
    with arcpy.da.UpdateCursor(layer, [field]) as cursor:
        for row in cursor:
            if np.sum(old_values == row[0]) > 0:
                row[0] = new_value
            cursor.updateRow(row)

#error in this function
def create_spill(layer: str, instructions: dict[str, any], grid_vector: str) -> str:
    """
    Creates geometries for oil spills

    Arguments:
        layer (str): feature class name
        instructions (dict): properties of operation to be performed

    Returns:
        layer_spill (str): spill polygon feature class name
    """
    aprint('Creating spill areas...')
    layer_spill = layer + '_spill'
    attr = 'HA_value'
    if instructions['type'] == 'oil':
        if not field_exists(layer, 's_vol'):
            arcpy.AddError(f'ERROR. Mandatory field "s_vol" does not exist in the input dataset.')
            delete_all_fc()
            exit()
        if not field_exists(layer, 's_area'):
            arcpy.AddError(f'ERROR. Mandatory field "s_area" does not exist in the input dataset.')
            delete_all_fc()
            exit()
        # replace 99999 and 0 values with None
        replace_layer_values(layer, 's_vol', [99999, 0], None)
        replace_layer_values(layer, 's_area', [99999, 0], None)
        # identify default volume and area values
        arr_vol = np.array([row[0] for row in arcpy.da.UpdateCursor(layer, ['s_vol'])], dtype=np.float32)
        default_vol = np.nanmedian(arr_vol)
        if np.isnan(default_vol):
            arcpy.AddWarning(f'WARNING. All "s_vol" values are empty for the dataset. No processing can be performed.')
            return None
        arr_area = np.array([row[0] for row in arcpy.da.UpdateCursor(layer, ['s_area'])], dtype=np.float32)
        default_area = np.nanmedian(arr_area)
        if np.isnan(default_area):
            arcpy.AddWarning(f'WARNING. All "s_area" values are empty for the dataset. No processing can be performed.')
            return None
        # create a field for buffer distance (= area radius) and new ones for area and value
        arcpy.AddField_management(in_table=layer, field_name='radius', field_type='FLOAT')
        arcpy.AddField_management(in_table=layer, field_name='area', field_type='FLOAT')
        arcpy.AddField_management(in_table=layer, field_name='volume', field_type='FLOAT')
        # calculate buffer distance (radius) for each row
        # if spill area <= 1 km2 then radius is super small
        with arcpy.da.UpdateCursor(layer, ['s_area', 'radius', 'area', 's_vol', 'volume']) as cursor:
            for row in cursor:
                area = default_area if row[0] == None or np.isnan(row[0]) else row[0]
                area = area * 1000000   # convert km2 to m2
                row[1] = float(np.sqrt(area / np.pi)) if area > 1000000 else 1
                row[2] = float(area)
                row[4] = float(default_vol) if row[3] == None or np.isnan(row[3]) else row[3]
                cursor.updateRow(row)
        # create buffers and set the HA value
        temp_spill = 'temp_spill'
        arcpy.Buffer_analysis(layer, temp_spill, 'radius')
        arcpy.AddField_management(in_table=temp_spill, field_name=attr, field_type='FLOAT')
        with arcpy.da.UpdateCursor(temp_spill, ['volume', 'area', attr]) as cursor:
            for row in cursor:
                row[2] = float(row[0] / (row[1] / 1000000)) if row[1] > 1000000 else row[0]
                cursor.updateRow(row)
    elif instructions['type'] == 'ship':
        if not field_exists(layer, 's_vol'):
            arcpy.AddError(f'ERROR. Mandatory field "s_vol" does not exist in the input dataset.')
            delete_all_fc()
            exit()
        # replace 99999 and 0 values with None
        replace_layer_values(layer, 's_vol', [99999, 0], None)
        # identify default volume
        arr_vol = np.array([row[0] for row in arcpy.da.UpdateCursor(layer, ['s_vol'])], dtype=np.float32)
        default_vol = np.nanmedian(arr_vol)
        if np.isnan(default_vol):
            arcpy.AddWarning(f'WARNING. All "s_vol" values are empty for the dataset. No processing can be performed.')
            return None
        # set HA value
        temp_spill = 'temp_spill'
        arcpy.CopyFeatures_management(layer, temp_spill)
        arcpy.AddField_management(in_table=temp_spill, field_name=attr, field_type='FLOAT')
        with arcpy.da.UpdateCursor(temp_spill, ['s_vol', attr]) as cursor:
            for row in cursor:
                # set spill volume
                row[1] = float(default_vol) if row[0] == None or np.isnan(row[0]) else row[0]
                cursor.updateRow(row)
    else:
        arcpy.AddWarning(f'WARNING. Unknown spill type: {instructions["type"]} is given in the configuration file. No processing can be performed.')
        return None
    # create layer for the grid
    temp_grid = 'temp_grid'
    arcpy.CopyFeatures_management(grid_vector, temp_grid)
    # create fieldmapping to set 'HA_value' attribute to be summed by spatial join
    fieldmappings = arcpy.FieldMappings()
    fieldmappings.addTable(temp_spill)
    field_index = fieldmappings.findFieldMapIndex(attr)
    fieldmap = fieldmappings.getFieldMap(field_index)
    fieldmap.mergeRule = 'sum'
    fieldmappings.replaceFieldMap(field_index, fieldmap)
    # create spatial join to sum together values of 'HA_value' field into grid layer
    arcpy.SpatialJoin_analysis(target_features=temp_grid, 
                            join_features=temp_spill, 
                            out_feature_class=layer_spill, 
                            join_operation='JOIN_ONE_TO_ONE', 
                            join_type='KEEP_ALL', 
                            field_mapping=fieldmappings, 
                            match_option='INTERSECT')
    with arcpy.da.UpdateCursor(layer_spill, [attr]) as cursor:
        for row in cursor:
            if row[0] == None:
                row[0] = 0
            cursor.updateRow(row)
    for fc in [temp_spill, temp_grid]:
        delete_fc(fc)
    return layer_spill

#checked
def create_birds(layer: str, instructions: dict[str, any], study_area: str, eez_borders: str) -> str:
    """
    Creates geometries for bird layer

    Arguments:
        layer (str): feature class name
        instructions (dict): properties of operation to be performed
        study_area (str): name of study area feature class
        eez_borders (str): name of exclusive economic zone feature class

    Returns:
        layer_birds (str): buffered county polygon feature class name
    """
    aprint('Creating county polygons...')
    layer_birds = layer + '_birds'
    attr = 'HA_value'

    # create a new layer for calculations
    aprint('\tCreating new feature class for dissolved geometries...')
    temp_dissolve = 'temp_dissolve'
    arcpy.CreateFeatureclass_management(arcpy.Describe(layer).path, temp_dissolve, 'POLYGON', layer)
    # add HA value field to new layer
    arcpy.AddField_management(temp_dissolve, attr, 'FLOAT')

    # identify all geometries and save the data from them
    aprint('\tIdentifying unique geometries...')
    geometries = {}
    if not field_exists(layer, 'value'):
        arcpy.AddError(f'ERROR. Mandatory field "value" does not exist in the input dataset.')
        delete_all_fc()
        exit()
    if not field_exists(layer, 'year'):
        arcpy.AddError(f'ERROR. Mandatory field "year" does not exist in the input dataset.')
        delete_all_fc()
        exit()
    with arcpy.da.SearchCursor(layer, ['SHAPE@', 'year', 'value']) as cursor:
        for row in cursor:
            geom = row[0].WKT
            year = row[1]
            value = row[2]
            if geom not in geometries:
                geometries[geom] = {}
            if year not in geometries[geom]:
                geometries[geom][year] = []
            geometries[geom][year].append(value)
    aprint(f'\tFound {len(list(geometries.keys()))} geometries')

    # create a new row for each geometry
    # give it the value of the yearly average of the sum all species
    aprint('\tMerging overlaying polygons...')
    with arcpy.da.InsertCursor(temp_dissolve, ['SHAPE@', attr]) as cursor:
        for geom, years in geometries.items():
            annual_sum = []
            for year, values in years.items():
                annual = 0
                for value in values:
                    annual += value
                annual_sum.append(annual)
            avg = sum(annual_sum) / len(annual_sum)
            cursor.insertRow([arcpy.FromWKT(geom), avg])
    
    # add a field for county id
    arcpy.AddField_management(temp_dissolve, 'county_id', 'LONG')
    # link OID to county id to HA value
    county_id_attr = {}
    with arcpy.da.UpdateCursor(temp_dissolve, ['OID@', 'county_id', attr]) as cursor:
        for row in cursor:
            row[1] = row[0]
            cursor.updateRow(row)
            county_id_attr[row[0]] = row[2]
    
    # erase marine features (= overlapping with study area)
    aprint('\tErasing marine areas...')
    temp_erase = 'temp_erase'
    arcpy.Erase_analysis(temp_dissolve, study_area, temp_erase)

    # create separate temporary layers for each country
    aprint('\tIdentifying country borders...')
    country_borders = []
    country_ids = []
    with arcpy.da.SearchCursor(eez_borders, ['OID@']) as cursor:
        for row in cursor:
            country_ids.append(row[0])
    for id in country_ids:
        # select current country
        temp_query_layer = f'temp_query_layer_{id}'
        arcpy.MakeFeatureLayer_management(eez_borders, temp_query_layer)
        oid = arcpy.Describe(temp_query_layer).OIDFieldName
        search_string = f'{oid} = {id}'
        arcpy.SelectLayerByAttribute_management(temp_query_layer, 'NEW_SELECTION', search_string)
        # create new feature class for country
        temp_country = f'temp_country_{id}'
        arcpy.CopyFeatures_management(temp_query_layer, temp_country)
        country_borders.append(temp_country)
        # clear selection
        arcpy.SelectLayerByAttribute_management(temp_query_layer, selection_type='CLEAR_SELECTION')
        # check that only one country was selected
        assert int(arcpy.GetCount_management(temp_country)[0]) == 1

    # extend non-water-intersecting polygons into water for each country
    has_counties = {}
    aprint('\tExtending county marine borders...')
    for country, id in zip(country_borders, country_ids):
        aprint(f'\t\tCountry id: {id}')
        # select counties within country
        aprint('\t\t\tSelecting counties within country...')
        temp_country_counties = f'temp_country_counties_{id}'
        arcpy.Intersect_analysis([temp_erase, country], temp_country_counties)
        # if country has no county polygons, skip
        has_counties[id] = int(arcpy.GetCount_management(temp_country_counties)[0]) > 0
        if not has_counties[id]:
            aprint('\t\t\tNo counties within country.')
            continue
        # do euclidean allocation on selection to extend polygons into sea
        aprint('\t\t\tPerforming euclidean allocation...')
        temp_eucal = f'temp_eucal_{id}'
        cell_size = 100
        arcpy.sa.EucAllocation(temp_country_counties, maximum_distance=instructions['buffer'], 
                            cell_size=cell_size, source_field='county_id').save(temp_eucal)
        # convert raster back to polygon
        aprint('\t\t\tConverting to polygons...')
        temp_buffer = f'temp_buffer_{id}'
        arcpy.RasterToPolygon_conversion(temp_eucal, temp_buffer, 'NO_SIMPLIFY')
        # dissolve same counties by gridcode
        temp_dissolve_counties = f'temp_dissolve_counties_{id}'
        arcpy.Dissolve_management(temp_buffer, temp_dissolve_counties, 'gridcode', multi_part='MULTI_PART')
        # link county id back to HA value (gridcode = county id)
        arcpy.AddField_management(temp_dissolve_counties, attr, 'FLOAT')
        with arcpy.da.UpdateCursor(temp_dissolve_counties, ['gridcode', attr]) as cursor:
            for row in cursor:
                row[1] = county_id_attr[row[0]]
                cursor.updateRow(row)
        # remove created areas outside of country borders
        aprint('\t\t\tClipping features to country borders...')
        temp_birds = f'temp_birds_{id}'
        arcpy.Clip_analysis(temp_dissolve_counties, country, temp_birds)
    # merge country-county layers back to one
    aprint('\t\tMerging country layers...')
    arcpy.Merge_management([f'temp_birds_{id}' for id in country_ids if has_counties[id]], layer_birds)
    # delete temporary layers
    for fc in [temp_dissolve, temp_erase]:
        delete_fc(fc)
    for x in ['temp_query_layer_', 'temp_country_', 'temp_country_counties_', 'temp_eucal_', 'temp_buffer_', 'temp_dissolve_counties_', 'temp_birds_']:
        for id in country_ids:
            if arcpy.Exists(x + f'{id}'):
                delete_fc(x + f'{id}')
    
    return layer_birds

#checked
def create_mammals(layer: str, instructions: dict[str, any], study_area: str, eez_borders: str) -> str:
    """
    Creates geometries for mammal layer

    Arguments:
        layer (str): feature class name
        instructions (dict): properties of operation to be performed
        study_area (str): name of study area feature class
        eez_borders (str): name of exclusive economic zone feature class

    Returns:
        layer_mammals (str): buffered county polygon feature class name
    """
    aprint('Creating mammal polygons...')
    layer_mammals = layer + '_mammals'
    attr = 'HA_value'
    value_attr = instructions['attribute_value']
    year_attr = instructions['attribute_year']
    area_attr = instructions['attribute_area']
    quota_attr = instructions['attribute_quota']
        
    # get all dataset attributes to the list
    dataset_attributes = [f.name for f in arcpy.ListFields(layer)]
    
    # check if quota attribute exists
    quota_exists = quota_attr in dataset_attributes

    # validate quota values, if all values are null then quota_exists is false
    if quota_exists:
        if not field_exists(layer, quota_attr):
            arcpy.AddError(f'ERROR. Mandatory field "{quota_attr}" does not exist in the input dataset.')
            delete_all_fc()
            exit()
        all_values_null = True
        with arcpy.da.SearchCursor(layer, quota_attr) as cursor:
            for row in cursor:
                if row[0] != None:
                    all_values_null = False
        if all_values_null:
            quota_exists = False
    
    # if quota_exists, make sure there are no null values (change them to zero)
    if quota_exists:
        with arcpy.da.UpdateCursor(layer, quota_attr) as cursor:
            for row in cursor:
                if row[0] == None:
                    row[0] = 0
                cursor.updateRow(row)
    
    # create a new layer for calculations
    aprint('\tCreating new feature class for dissolved geometries...')
    temp_dissolve = 'temp_dissolve'
    arcpy.CreateFeatureclass_management(arcpy.Describe(layer).path, temp_dissolve, 'POLYGON', layer)
    # add HA value field to new layer
    arcpy.AddField_management(temp_dissolve, attr, 'FLOAT')

    # identify all geometries and save the data from them
    aprint('\tIdentifying unique geometries...')
    if not field_exists(layer, value_attr):
        arcpy.AddError(f'ERROR. Mandatory field "{value_attr}" does not exist in the input dataset.')
        delete_all_fc()
        exit()   
    if not field_exists(layer, year_attr):
        arcpy.AddError(f'ERROR. Mandatory field "{year_attr}" does not exist in the input dataset.')
        delete_all_fc()
        exit()
    if not field_exists(layer, area_attr):
        arcpy.AddError(f'ERROR. Mandatory field "{area_attr}" does not exist in the input dataset.')
        delete_all_fc()
        exit()
    fields = ['SHAPE@', year_attr, value_attr, area_attr]
    if quota_exists: fields += [quota_attr]
    geometries = {}
    with arcpy.da.SearchCursor(layer, fields) as cursor:
        for row in cursor:
            #geom = row[0].WKT
            geom = row[0]
            year = row[1]
            value = row[2]
            area = row[3]
            quota = row[-1] if quota_exists else None
            if value != None:
                if area not in geometries:
                    geometries[area] = {
                        "values": [],
                        "geometry": geom.WKT
                    }
                # calculate value differently depending on if quota is known
                if quota != None:
                    if quota == 0:
                        if value == 0:
                            # if both quota and value are zero, give value 0
                            val = 0
                        else:
                            # if only quota is zero, give value 0.5
                            val = 0.5
                    else:
                        # divide by quota and multiply with 0.5
                        val = value / quota * 0.5
                else:
                    # if no quota, give value just like that (previously normalized)
                    val = value
                geometries[area]["values"].append(val)
    aprint(f'\tFound {len(list(geometries.keys()))} geometries')

    # create a new row for each geometry
    # give it the value of the yearly average of the sum all species (but for mammals only one species)
    aprint('\tMerging overlaying polygons...')
    with arcpy.da.InsertCursor(temp_dissolve, ['SHAPE@', attr]) as cursor:
        for area, entry in geometries.items():
            avg = sum(entry["values"]) / len(entry["values"])
            cursor.insertRow([arcpy.FromWKT(entry["geometry"]), avg])
    
    # add a field for county id
    arcpy.AddField_management(temp_dissolve, 'area_id', 'LONG')
    # link OID to county id to HA value
    county_id_attr = {}
    with arcpy.da.UpdateCursor(temp_dissolve, ['OID@', 'area_id', attr]) as cursor:
        for row in cursor:
            row[1] = row[0]
            cursor.updateRow(row)
            county_id_attr[row[0]] = row[2]
    
    # erase marine features (= overlapping with study area)
    aprint('\tErasing marine areas...')
    temp_erase = 'temp_erase'
    arcpy.Erase_analysis(temp_dissolve, study_area, temp_erase)

    # create separate temporary layers for each country
    aprint('\tIdentifying country borders...')
    country_borders = []
    country_ids = []
    with arcpy.da.SearchCursor(eez_borders, ['OID@']) as cursor:
        for row in cursor:
            country_ids.append(row[0])
    for id in country_ids:
        # select current country
        temp_query_layer = f'temp_query_layer_{id}'
        arcpy.MakeFeatureLayer_management(eez_borders, temp_query_layer)
        oid = arcpy.Describe(temp_query_layer).OIDFieldName
        search_string = f'{oid} = {id}'
        arcpy.SelectLayerByAttribute_management(temp_query_layer, 'NEW_SELECTION', search_string)
        # create new feature class for country
        temp_country = f'temp_country_{id}'
        arcpy.CopyFeatures_management(temp_query_layer, temp_country)
        country_borders.append(temp_country)
        # clear selection
        arcpy.SelectLayerByAttribute_management(temp_query_layer, selection_type='CLEAR_SELECTION')
        # check that only one country was selected
        assert int(arcpy.GetCount_management(temp_country)[0]) == 1

    # extend non-water-intersecting polygons into water for each country
    has_counties = {}
    aprint('\tExtending county marine borders...')
    temp_selectbyloc_layer = f'temp_selectbyloc_layer'
    arcpy.MakeFeatureLayer_management(temp_erase, temp_selectbyloc_layer)
    for country, id in zip(country_borders, country_ids):
        aprint(f'\t\tCountry id: {id}')
        # select counties within country
        aprint('\t\t\tSelecting counties within country...')
        arcpy.management.SelectLayerByLocation(temp_selectbyloc_layer, "have_their_center_in", country)
        temp_country_counties = f'temp_country_counties_{id}'
        arcpy.management.CopyFeatures(temp_selectbyloc_layer, temp_country_counties)
        arcpy.SelectLayerByAttribute_management(temp_selectbyloc_layer, selection_type='CLEAR_SELECTION')
        #arcpy.Intersect_analysis([temp_erase, country], temp_country_counties)
        # if country has no county polygons, skip
        has_counties[id] = int(arcpy.GetCount_management(temp_country_counties)[0]) > 0
        if not has_counties[id]:
            aprint('\t\t\tNo counties within country.')
            continue
        # do euclidean allocation on selection to extend polygons into sea
        aprint('\t\t\tPerforming euclidean allocation...')
        temp_eucal = f'temp_eucal_{id}'
        cell_size = 101
        arcpy.sa.EucAllocation(temp_country_counties, maximum_distance=instructions['buffer'], 
                            cell_size=cell_size, source_field='area_id').save(temp_eucal)
        # convert raster back to polygon
        aprint('\t\t\tConverting to polygons...')
        temp_buffer = f'temp_buffer_{id}'
        arcpy.RasterToPolygon_conversion(temp_eucal, temp_buffer, 'NO_SIMPLIFY')
        # dissolve same counties by gridcode
        temp_dissolve_counties = f'temp_dissolve_counties_{id}'
        arcpy.Dissolve_management(temp_buffer, temp_dissolve_counties, 'gridcode', multi_part='MULTI_PART')
        # link county id back to HA value (gridcode = county id)
        arcpy.AddField_management(temp_dissolve_counties, attr, 'FLOAT')
        with arcpy.da.UpdateCursor(temp_dissolve_counties, ['gridcode', attr]) as cursor:
            for row in cursor:
                row[1] = county_id_attr[row[0]]
                cursor.updateRow(row)
        # remove created areas outside of country borders
        aprint('\t\t\tClipping features to country borders...')
        temp_birds = f'temp_mammals_{id}'
        arcpy.Clip_analysis(temp_dissolve_counties, country, temp_birds)
    # merge country-county layers back to one
    aprint('\t\tMerging country layers...')
    arcpy.Merge_management([f'temp_mammals_{id}' for id in country_ids if has_counties[id]], layer_mammals)
    # delete temporary layers
    for fc in [temp_dissolve, temp_erase]:
        delete_fc(fc)
    for x in ['temp_query_layer_', 'temp_country_', 'temp_country_counties_', 'temp_eucal_', 'temp_buffer_', 'temp_dissolve_counties_', 'temp_birds_']:
        for id in country_ids:
            if arcpy.Exists(x + f'{id}'):
                delete_fc(x + f'{id}')
    
    return layer_mammals

#checked
def create_fish_original(layer: str, effort: str, instructions: dict[str, any]) -> str:
    """
    Creates geometries for fish layer

    Arguments:
        layer (str): landings feature class name
        effort (str): fishing effort feature class name
        instructions (dict): properties of operation to be performed

    Returns:
        layer_fish (str): output polygon feature class name
    """
    aprint('Creating fish polygons...')
    layer_fish = layer + '_fish'
    attr = 'HA_value'

    aprint('\tCopying effort layer...')
    effort_copy = 'temp_effort'
    arcpy.CopyFeatures_management(effort, effort_copy)

    #
    # preprocess landing layer
    #

    # find value fields in layer and merge into one
    fields = [f.name for f in arcpy.ListFields(layer, instructions['landing_attribute'] + '*')]
    landing_attr = 'landing_attr'
    objectid = 'OBJECTID'
    sq_ID = 'sq_ID'
    # Final attribute in the big polygon layer which store value for calculating HA value
    arcpy.AddField_management(layer, landing_attr, 'FLOAT')
    # Attribute to store big polygon ID for grouping small polygons within big
    arcpy.AddField_management(layer, sq_ID, 'TEXT')
    with arcpy.da.UpdateCursor(layer, fields + [landing_attr, objectid, sq_ID]) as cursor:
        for row in cursor:
            values = [val for val in row[:-3] if val != None]
            mean = sum(values) / len(values) if len(values) > 0 else None
            row[-3] = mean
            row[-1] = str(row[-2])
            cursor.updateRow(row)
    
    # create a new layer for landing on which to merge overlapping polygons
    aprint('\tCreating new feature class for dissolved landing geometries...')
    landing_dissolve = 'temp_landing_dissolve'
    arcpy.CreateFeatureclass_management(arcpy.Describe(layer).path, landing_dissolve, 'POLYGON', layer)
                
    aprint('\tIdentifying unique geometries...')
    geometries = {}
    with arcpy.da.SearchCursor(layer, ['SHAPE@', landing_attr, sq_ID]) as cursor:
        for row in cursor:
            geom = row[0].WKT
            value = row[1]
            if value != None:
                if geom not in geometries:
                    geometries[geom] = {
                        "sq_id": row[2],
                        "val": []
                    }
                geometries[geom]["val"].append(value)
    aprint(f'\tFound {len(list(geometries.keys()))} unique geometries out of {int(arcpy.GetCount_management(layer)[0])}')
    
    # create a new row for each geometry
    # give it the value of the average
    aprint('\tMerging overlaying polygons...')
    with arcpy.da.InsertCursor(landing_dissolve, ['SHAPE@', landing_attr, sq_ID]) as cursor:
        for geom, squares in geometries.items():
            new_value = np.mean(squares["val"])
            cursor.insertRow([arcpy.FromWKT(geom), new_value, squares["sq_id"]])

    #
    # preprocess effort layer
    #
    
    # create a new layer for effort on which to merge overlapping polygons
    aprint('\tCreating new feature class for dissolved effort geometries...')
    effort_dissolve = 'temp_effort_dissolve'
    arcpy.CreateFeatureclass_management(arcpy.Describe(effort_copy).path, effort_dissolve, 'POLYGON', effort_copy)
    # add effort value field to new layer
    effort_attr = 'effort_attr'
    arcpy.AddField_management(effort_dissolve, effort_attr, 'FLOAT')

    # identify all geometries and save the data from them
    aprint('\tIdentifying unique geometries...')
    geometries = {}
    if not field_exists(effort_copy, 'year'):
        arcpy.AddError(f'ERROR. Mandatory field "year" does not exist in the input dataset.')
        delete_all_fc()
        exit()
    check_field = instructions['effort_attribute']
    if not field_exists(effort_copy, check_field):
        arcpy.AddError(f'ERROR. Mandatory field "{check_field}" does not exist in the input dataset.')
        delete_all_fc()
        exit()
    with arcpy.da.SearchCursor(effort_copy, ['SHAPE@', 'year', instructions['effort_attribute']]) as cursor:
        for row in cursor:
            geom = row[0].WKT
            year = row[1]
            value = row[2]
            if value != None:
                if geom not in geometries:
                    geometries[geom] = {}
                if year not in geometries[geom]:
                    geometries[geom][year] = []
                if value != None:
                    geometries[geom][year].append(value)
    aprint(f'\tFound {len(list(geometries.keys()))} unique geometries out of {int(arcpy.GetCount_management(effort_copy)[0])}')

    # create a new row for each geometry
    # give it the value of the yearly average of the sum all species (but for mammals only one species)
    aprint('\tMerging overlaying polygons...')
    with arcpy.da.InsertCursor(effort_dissolve, ['SHAPE@', effort_attr]) as cursor:
        for geom, years in geometries.items():
            annual_sum = []
            for year, values in years.items():
                annual_sum.append(sum([v for v in values if v != None]))
            if len(annual_sum) > 0:
                avg = sum(annual_sum) / len(annual_sum)
                cursor.insertRow([arcpy.FromWKT(geom), avg])

    #
    # actual processing of the layer
    #
    
    # Create a spatial join layer. Join large polygon attributes to small polygon. 
    aprint('\tSpatial join large polygons to small...')
    fieldmappings = arcpy.FieldMappings()
    fieldmappings.addTable(effort_dissolve)
    fieldmappings.addTable(layer)
    
    arcpy.analysis.SpatialJoin(effort_dissolve, layer, layer_fish, "JOIN_ONE_TO_ONE", "KEEP_COMMON", fieldmappings, "LARGEST_OVERLAP")
    
    # Sum effort values from all small polygons within each big polygon
    aprint('\tCalculate HA values for small polygons...')
    bigsq = {}
    with arcpy.da.SearchCursor(layer_fish, [sq_ID, landing_attr, effort_attr]) as cursor:
        for row in cursor:
            fid = row[0]
            if row[1] != None: 
                if fid not in bigsq:
                    bigsq[fid] = {
                        "landing_val": row[1],
                        "effort_sum": row[2]
                    }
                else:
                    bigsq[fid]["effort_sum"] += row[2]
    
    # add HA value field to layer
    arcpy.AddField_management(layer_fish, attr, 'FLOAT')
             
    # calculate HA values
    # set value as small square value
    # divide by sum of small square values within large squares
    # multiply with large square value
    # divide by area (converted to m2)
    with arcpy.da.UpdateCursor(layer_fish, [attr, effort_attr, sq_ID, 'Shape_Area']) as cursor:
        for row in cursor:
            fid = row[2]
            if bigsq[fid]["effort_sum"] == 0:
                row[0] = 0
            else:
                row[0] = row[1] / bigsq[fid]["effort_sum"] * bigsq[fid]["landing_val"] / (row[3] / 1000000)
            cursor.updateRow(row)
    
    # # create intersect layer between landings and effort to not have effort polygons crossing landing borders
    # aprint('\tIntersecting small and large polygons...')
    # temp_intersect = 'temp_intersect'
    # arcpy.Intersect_analysis([landing_dissolve, effort_dissolve], temp_intersect)
    # # sum small square attribute values to corresponding large squares
    # aprint('\tSumming small polygon values inside large polygons...')
    # temp_large_grid_sum = 'temp_large_grid_sum'
    # arcpy.SummarizeWithin_analysis(landing_dissolve, temp_intersect, temp_large_grid_sum, 'KEEP_ALL', [[effort_attr, 'SUM']])
    # # assign same summed values back to the small squares by doing intersect again
    # # temp_intersect should after that contain small square value, large square value, and sum of small square values within large squares
    # aprint('\tIntersecting again...')
    # arcpy.Intersect_analysis([temp_large_grid_sum, effort_dissolve], layer_fish)
    # # create HA value field
    # # set value as small square value
    # # divide by sum of small square values within large squares
    # # multiply with large square value
    # # divide by area (converted to m2)
    # aprint('\tCalculating HA values...')
    # arcpy.AddField_management(layer_fish, attr, 'FLOAT')
    # with arcpy.da.UpdateCursor(layer_fish, [attr, 'sum_'+effort_attr, effort_attr, landing_attr, 'Shape_Area']) as cursor:
        # for row in cursor:
            # if row[1] == 0:
                # row[0] = 0
            # else:
                # row[0] = row[2] / row[1] * row[3] / (row[4] / 1000000)
            # cursor.updateRow(row)
    # for fc in [effort_copy, temp_select, temp_intersect, temp_large_grid_sum, effort_dissolve, landing_dissolve]:
        # delete_fc(fc)
    return layer_fish
#checked    
def create_fish_alternative(layer: str, instructions: dict[str, any]) -> str:
    """
    Creates geometries for fish layer

    Arguments:
        layer (str): landings feature class name
        instructions (dict): properties of operation to be performed

    Returns:
        layer_fish (str): output polygon feature class name
    """
    aprint('Creating fish polygons...')
    layer_fish = layer + '_fish'
    attr = 'HA_value'

    # create a new layer on which to merge overlapping polygons
    aprint('\tCreating new feature class for dissolved geometries...')
    temp_dissolve = 'temp_dissolve'
    arcpy.CreateFeatureclass_management(arcpy.Describe(layer).path, layer_fish, 'POLYGON', layer)
    arcpy.AddField_management(layer_fish, attr, 'FLOAT')
                
    aprint('\tIdentifying unique geometries...')
    geometries = {}
    sq_ID_attr = 'cscode'
    year_attr = 'year'
    value_attr = 'ttfshdy'
    if not field_exists(layer, sq_ID_attr):
        arcpy.AddError(f'ERROR. Mandatory field "{sq_ID_attr}" does not exist in the input dataset.')
        delete_all_fc()
        return
    if not field_exists(layer, year_attr):
        arcpy.AddError(f'ERROR. Mandatory field "{year_attr}" does not exist in the input dataset.')
        delete_all_fc()
        return
    if not field_exists(layer, value_attr):
        arcpy.AddError(f'ERROR. Mandatory field "{value_attr}" does not exist in the input dataset.')
        delete_all_fc()
        return
    with arcpy.da.SearchCursor(layer, ['SHAPE@', sq_ID_attr, year_attr, value_attr]) as cursor:
        for row in cursor:
            sq_ID = row[1]
            year = row[2]
            value = row[3]
            if sq_ID not in geometries:
                geometries[sq_ID] = {
                    'geom': row[0].WKT
                }
            if value != None:
                if year not in geometries[sq_ID]:
                    geometries[sq_ID][year] = value
                else:
                    geometries[sq_ID][year] += value
                
    aprint(f'\tFound {len(list(geometries.keys()))} unique geometries out of {int(arcpy.GetCount_management(layer)[0])}')
    
    aprint('\tMerging overlaying polygons...')
    with arcpy.da.InsertCursor(layer_fish, ['SHAPE@', attr]) as cursor:
        for sq_ID, values in geometries.items():
            annual_sum = 0
            cnt = 0
            wkt = None
            for key, value in values.items():
                if key == 'geom':
                    wkt = value
                else:
                    annual_sum += value
                    cnt += 1
            if cnt > 0 and wkt:
                avg = annual_sum / cnt
                cursor.insertRow([arcpy.FromWKT(wkt), avg])
                
    with arcpy.da.UpdateCursor(layer_fish, [attr, 'Shape_Area']) as cursor:
        for row in cursor:
            row[0] = row[0] / (row[1] / 1000000)
            cursor.updateRow(row)
    
    return layer_fish

#checked
def create_rivers(layer: str, rivers: str, instructions: dict[str, any]) -> str:
    """
    Creates geometries for rivers layer

    Arguments:
        layer (str): hydropower dams feature class name
        rivers (str): rivers feature class name
        instructions (dict): properties of operation to be performed

    Returns:
        layer_rivers (str): buffered river polygon feature class name
    """
    aprint('Creating dam intersecting rivers...')
    # buffer rivers and dissolve them in case of multi segment rivers
    temp_rivers_buffer = 'temp_rivers_buffer'
    arcpy.Buffer_analysis(rivers, temp_rivers_buffer, instructions['buffer'])
    temp_rivers_dissolve = 'temp_rivers_dissolve'
    arcpy.Dissolve_management(temp_rivers_buffer, temp_rivers_dissolve, multi_part='SINGLE_PART')
    # select rivers close to hydropower dams
    temp_rivers = 'temp_rivers'
    arcpy.MakeFeatureLayer_management(temp_rivers_dissolve, temp_rivers)
    arcpy.SelectLayerByLocation_management(temp_rivers, 'WITHIN_A_DISTANCE', layer, instructions['distance'], 'NEW_SELECTION')
    layer_rivers = layer + '_rivers'
    arcpy.CopyFeatures_management(temp_rivers, layer_rivers)
    arcpy.SelectLayerByAttribute_management(temp_rivers, selection_type='CLEAR_SELECTION')
    for fc in [temp_rivers, temp_rivers_buffer, temp_rivers_dissolve]:
        delete_fc(fc)
    return layer_rivers

#checked
def sum_overlapping(layer_in: str, layer_out: str, attr: str):
    """
    Sums given attribute values of overlapping polygons
    """
    aprint('\tSumming overlapping polygons...')
    # create new buffer polygons just slightly larger than the layer polygons
    aprint('\t\tCreating join features...')
    temp_buffer = 'temp_sum_overlap_buffer'
    arcpy.Buffer_analysis(layer_in, temp_buffer, '1 Meters')
    # split every polygon overlap into its own feature
    aprint('\t\tCreating target features...')
    temp_poly = 'temp_sum_overlap_poly'
    arcpy.FeatureToPolygon_management(layer_in, temp_poly)
    # create fieldmapping to set attribute to be summed by spatial join
    aprint('\t\tCreating field mapping...')
    fieldmappings = arcpy.FieldMappings()
    fieldmappings.addTable(temp_buffer)     # changed from temp_poly
    field_index = fieldmappings.findFieldMapIndex(attr)
    fieldmap = fieldmappings.getFieldMap(field_index)
    fieldmap.mergeRule = 'sum'
    fieldmappings.replaceFieldMap(field_index, fieldmap)
    # create spatial join to sum together values of attr field into temp_poly layer
    aprint('\t\tPerforming spatial join...')
    arcpy.SpatialJoin_analysis(target_features=temp_poly, 
                                join_features=temp_buffer, 
                                out_feature_class=layer_out, 
                                join_operation='JOIN_ONE_TO_ONE', 
                                join_type='KEEP_ALL', 
                                field_mapping=fieldmappings, 
                                match_option='WITHIN')
    for fc in [temp_buffer, temp_poly]:
        delete_fc(fc)

#checked
def post_process(layer: str, instructions: list, layer_code: str, study_area: str) -> str:
    """
    Does postprocessing on geometries

    Arguments:
        layer (str): clipped feature class name
        instructions (list): post processing operations to be performed on processed geometries
        layer_code (str): unique layer identification code
        study_area (str): study area layer for clipping

    Returns:
        layer_processed (str): processed polygon feature class name
    """
    layer_processed = layer

    for operation in instructions:

        if operation == 'dissolve':
            aprint('\tDissolving overlapping features...')
            temp_dissolved = 'temp_dissolved_' + layer_code
            # since no fields are specified, all touching features are dissolved
            arcpy.Dissolve_management(layer_processed, temp_dissolved, multi_part='MULTI_PART')
            if layer_processed != layer: delete_fc(layer_processed)
            layer_processed = temp_dissolved
        
        elif operation == 'clip':
            aprint('\tClipping to borders...')
            temp_clipped = 'temp_clipped_' + layer_code
            arcpy.Clip_analysis(layer_processed, study_area, temp_clipped)
            if layer_processed != layer: delete_fc(layer_processed)
            layer_processed = temp_clipped

        elif operation == 'sum_overlap':
            #sum together overlapping polygon values of 'HA_value' field
            temp_sum_overlap = 'temp_sum_overlap_' + layer_code
            sum_overlapping(layer_processed, temp_sum_overlap, 'HA_value')
            # make sure there are no rows with None values
            with arcpy.da.UpdateCursor(temp_sum_overlap, ['HA_value']) as cursor:
                for row in cursor:
                    if row[0] == None:
                        cursor.deleteRow()
            if layer_processed != layer: delete_fc(layer_processed)
            layer_processed = temp_sum_overlap
            
        elif operation == 'max_one':
            aprint('\tSetting max value to 1...')
            temp_max_one = 'temp_max_one_' + layer_code
            arcpy.CopyFeatures_management(layer_processed, temp_max_one)
            # make sure all HA values are below 1
            with arcpy.da.UpdateCursor(temp_max_one, ['HA_value']) as cursor:
                for row in cursor:
                    row[0] = row[0] if row[0] <= 1 else 1
                    cursor.updateRow(row)
            if layer_processed != layer: delete_fc(layer_processed)
            layer_processed = temp_max_one
        
        elif operation == 'divide_by_area':
            aprint('\tDividing by area...')
            temp_divide_by_area = 'temp_divide_by_area_' + layer_code
            arcpy.CopyFeatures_management(layer_processed, temp_divide_by_area)
            # divide number of birds with marine area (km2) of county polygons (= the area that is left after clipping)
            with arcpy.da.UpdateCursor(temp_divide_by_area, ['Shape_Area', 'HA_value']) as cursor:
                for row in cursor:
                    row[1] = row[1] / (row[0] / 1000000)
                    cursor.updateRow(row)
            if layer_processed != layer: delete_fc(layer_processed)
            layer_processed = temp_divide_by_area

        elif operation == 'log_normalize':
            aprint('\tLog normalizing values...')
            temp_log_normalized = 'temp_log_normalized_' + layer_code
            arcpy.CopyFeatures_management(layer_processed, temp_log_normalized)
            with arcpy.da.UpdateCursor(temp_log_normalized, ['HA_value']) as cursor:
                for row in cursor:
                    row[0] = np.log10(row[0] + 1)
                    cursor.updateRow(row)
            layer_processed = temp_log_normalized

        elif operation == '':
            aprint('\tWARNING. Empty operation given. Skipping...')
            pass

        else:
            raise Exception(f'ERROR. Operation not recognized: {operation}')
    
    if len(instructions) == 0:
        layer_processed += '_postprocessed'
        arcpy.CopyFeatures_management(layer, layer_processed)

    return layer_processed

#checked
def normalize_vector(layer: str, field: str):
    """
    Normalizes values in the layer's given field so that min value is 0 and max is 1.
    Modifies original layer.
    """
    if not field_exists(layer, field):
        arcpy.AddError(f'ERROR. Mandatory field "{field}" does not exist in the input dataset.')
        delete_all_fc()
        exit()
    values = [x[0] for x in arcpy.da.SearchCursor(layer, [field])]
    min_val, max_val = min(values), max(values)
    diff = max_val - min_val
    with arcpy.da.UpdateCursor(layer, field) as cursor:
        for row in cursor:
            row[0] = (row[0] - min_val) / diff
            cursor.updateRow(row)

#checked
def normalize_raster(layer: str):
    """
    Normalizes values in the layer's given field so that min value is 0 and max is 1.
    Returns a new raster.
    """
    min_val = arcpy.GetRasterProperties_management(layer, "MINIMUM").getOutput(0)
    max_val = arcpy.GetRasterProperties_management(layer, "MAXIMUM").getOutput(0)
    min_val = min_val.replace(',', '.')
    max_val = max_val.replace(',', '.')
    diff = float(max_val) - float(min_val)
    normalized = arcpy.sa.Float((arcpy.sa.Raster(layer) - float(min_val)) / diff)
    return normalized

#checked
def create_raster(layer: str, instructions: list, layer_code: str, output: str, grid_vector: str, grid_raster: str, activity_rasters: list):
    """
    Rasterizes vector layer

    Arguments:
        layer (str): clipped feature class name
        instructions (list): operations to be performed to create output raster
        layer_code (str): unique layer identification code
        output (str): output raster name
        grid_vector (str): grid reference feature class
        grid_raster (str): grid reference raster
        activity_rasters (list): produced activity rasters used for pressure raster (if applicable)
    """
    aprint('Creating raster...')
    if arcpy.Exists(output):
        delete_fc(output)

    if activity_rasters == None: activity_rasters = [None]

    layer_processed = layer
    raster_processed = activity_rasters[0]
    
    for operation in instructions:

        if operation == 'copy':
            # make a copy of raster as the pressure output
            aprint('\tCopying raster...')
            temp_copy = 'temp_copy_' + layer_code
            arcpy.CopyRaster_management(raster_processed, temp_copy)
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_copy
    
        elif operation == 'area_coverage':
            # Select all cells which intersect with the layer polygons
            aprint('\tSelecting intersecting cells...')
            temp_select = 'temp_select_' + layer_code
            arcpy.MakeFeatureLayer_management(grid_vector, temp_select)
            arcpy.SelectLayerByLocation_management(temp_select, 'INTERSECT', layer_processed, selection_type='NEW_SELECTION')
            temp_grid = 'temp_grid_' + layer_code
            arcpy.CopyFeatures_management(temp_select, temp_grid)
            # create a new feature for each overlap between cells and layer polygons
            # since the features have been dissolved into one multipart, there will be only one feature per cell
            aprint('\tCounting overlapping features...')
            temp_grid_count = 'temp_grid_count_' + layer_code
            arcpy.CountOverlappingFeatures_analysis([temp_grid, layer_processed], temp_grid_count, 2)
            # remember to clear selection
            arcpy.SelectLayerByAttribute_management(temp_select, selection_type='CLEAR_SELECTION')
            # create raster with area value of largest polygon in cell
            # this value will be total polygon area in cell since there is only one polygon per cell
            aprint('\tRasterizing with area as value...')
            temp_raster = 'temp_raster_' + layer_code
            arcpy.PolygonToRaster_conversion(temp_grid_count, 'Shape_Area', temp_raster, 'MAXIMUM_AREA', 'Shape_Area', grid_raster)
            # divide cell value by 1 million (meters^2) to get cell area covered by polygon in km^2
            aprint('\tCalculating coverage...')
            temp_raster_calc = arcpy.sa.RasterCalculator([grid_raster, temp_raster], ['r1', 'r2'], 'r1 + Con(IsNull(r2), Float(0), r2/1000000)', 'FirstOf', 'FirstOf')
            temp_area_coverage = 'temp_raster_area_cover_' + layer_code
            temp_raster_calc.save(temp_area_coverage)
            for fc in [temp_grid_count, temp_grid, temp_raster, temp_select]:
                delete_fc(fc)
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_area_coverage

        elif operation == 'covered_cells':
            # create new grid polygons and add HA value field
            aprint('\tSetting all grid values to 0...')
            temp_grid = 'temp_grid_' + layer_code
            arcpy.CopyFeatures_management(grid_vector, temp_grid)
            arcpy.AddField_management(in_table=temp_grid, field_name='HA_value', field_type='SHORT')
            # give all cells default value 0
            with arcpy.da.UpdateCursor(in_table=temp_grid, field_names=['HA_value']) as cursor:
                for row in cursor:
                    row[0] = 0
                    cursor.updateRow(row)
            # select cells which intersect polygon layer, and give them value 1
            aprint('\tSelecting intersecting cells and setting value to 1...')
            temp_select = 'temp_select_' + layer_code
            arcpy.MakeFeatureLayer_management(temp_grid, temp_select)
            arcpy.SelectLayerByLocation_management(temp_select, 'INTERSECT', layer_processed, selection_type='NEW_SELECTION')
            with arcpy.da.UpdateCursor(in_table=temp_select, field_names=['HA_value']) as cursor:
                for row in cursor:
                    row[0] = 1
                    cursor.updateRow(row)
            # remember to clear selection
            arcpy.SelectLayerByAttribute_management(temp_select, selection_type='CLEAR_SELECTION')
            # convert grid to raster
            aprint('\tRasterizing...')
            temp_raster = 'temp_raster_' + layer_code
            arcpy.PolygonToRaster_conversion(temp_select, 'HA_value', temp_raster, 'MAXIMUM_AREA', None, grid_raster)
            for fc in [temp_grid, temp_select]:
                delete_fc(fc)
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_raster
            
        elif operation == 'sum_poly' or operation == 'sum_poly_log':
            # create intersect layer between grid and layer to not have layer polygons crossing cell borders
            aprint('\tIntersecting grid with polygons...')
            temp_intersect = 'temp_intersect_' + layer_code
            arcpy.Intersect_analysis([grid_vector, layer_processed], temp_intersect)
            # add HA value area field, and set value as HA value multiplied with polygon fraction of total cell area
            aprint('\tMultiplying polygon values with fraction of cell area...')
            arcpy.AddField_management(temp_intersect, 'HA_value_area', field_type='FLOAT')
            desc = arcpy.Describe(grid_raster)
            cell_area = desc.meanCellHeight * desc.meanCellWidth
            with arcpy.da.UpdateCursor(temp_intersect, ['Shape_area', 'HA_value', 'HA_value_area']) as cursor:
                for row in cursor:
                    row[2] = row[0] / cell_area * row[1]
                    cursor.updateRow(row)
            # create copy of grid and sum together fractional HA values
            aprint('\tSumming polygon values within grid cells...')
            temp_grid = 'temp_grid_' + layer_code
            arcpy.CopyFeatures_management(grid_vector, temp_grid)
            temp_grid_sum = 'temp_grid_sum_' + layer_code
            arcpy.SummarizeWithin_analysis(temp_grid, temp_intersect, temp_grid_sum, 'KEEP_ALL', [['HA_value_area', 'SUM']])
            # if necessary, log normalize values
            if operation == 'sum_poly_log':
                aprint('\tLog normalizing values...')
                with arcpy.da.UpdateCursor(temp_grid_sum, ['sum_HA_value_area']) as cursor:
                    for row in cursor:
                        row[0] = np.log10(row[0] + 1)
                        cursor.updateRow(row)
            # convert grid to raster
            aprint('\tRasterizing...')
            temp_raster = 'temp_raster_' + layer_code
            arcpy.PolygonToRaster_conversion(temp_grid_sum, 'sum_HA_value_area', temp_raster, 'MAXIMUM_AREA', 'sum_HA_value_area', grid_raster)
            for fc in [temp_intersect, temp_grid, temp_grid_sum]:
                delete_fc(fc)
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_raster
        
        elif operation == 'grid_to_raster':
            # convert polygon grid straight to raster
            aprint('\tRasterizing grid...')
            temp_raster = 'temp_raster_' + layer_code
            arcpy.PolygonToRaster_conversion(layer_processed, 'HA_value', temp_raster, 'MAXIMUM_AREA', None, grid_raster)
            raster_processed = temp_raster
        
        elif operation == 'poly_to_grid':
            # intersect polygons with grid
            aprint('\tIntersecting grid with polygons...')
            temp_intersect = 'temp_intersect_' + layer_code
            arcpy.Intersect_analysis([grid_vector, layer_processed], temp_intersect)
            # convert polygons to raster
            aprint('\tRasterizing grid intersected polygons...')
            temp_raster = 'temp_raster_' + layer_code
            arcpy.PolygonToRaster_conversion(temp_intersect, 'HA_value', temp_raster, None, 'HA_value', grid_raster)
            raster_processed = temp_raster
        
        elif operation == 'poly_to_grid_max_area':
            # convert polygons to raster
            aprint('\tRasterizing grid intersected polygons...')
            temp_raster = 'temp_raster_max_area_' + layer_code
            arcpy.PolygonToRaster_conversion(layer_processed, 'HA_value', temp_raster, 'MAXIMUM_AREA', None, grid_raster)
            raster_processed = temp_raster

        elif operation == 'raster_mean':
            aprint('\tCalculating mean of rasters...')
            temp_mean = 'temp_raster_mean_' + layer_code
            temp_raster = arcpy.ia.Mean(activity_rasters, 'UnionOf', 'MinOf', False)
            temp_raster.save(temp_mean)
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_mean
        
        elif operation == 'raster_sum':
            aprint('\tCalculating sum of rasters...')
            temp_sum = 'temp_raster_sum_' + layer_code
            temp_raster = arcpy.ia.Sum(activity_rasters, 'UnionOf', 'MinOf', True)
            temp_raster.save(temp_sum)
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_sum
            
        elif operation == 'normalize':
            aprint('\tNormalizing cell values...')
            temp_norm = 'temp_raster_normalized_' + layer_code
            normalize_raster(raster_processed).save(temp_norm)
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_norm
            
        elif operation == 'log10':
            aprint('\tConverting cell values to log10...')
            temp_log10 = 'temp_raster_log10_' + layer_code
            arcpy.sa.Log10(arcpy.Raster(raster_processed)).save(temp_log10)
            
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_log10
            
        elif operation == 'log10_value_plus_1':
            aprint('\tAdding 1 to values and converting cell values to log10...')
            raster_plus_1 = 'temp_raster_plus_1_' + layer_code
            r_plus_1 = arcpy.Raster(raster_processed) + 1
            r_plus_1.save(raster_plus_1)
            
            temp_log10 = 'temp_raster_log10_' + layer_code
            arcpy.sa.Log10(arcpy.Raster(raster_plus_1)).save(temp_log10)
            
            if raster_processed != activity_rasters[0]: delete_fc(raster_processed)
            raster_processed = temp_log10
        
        else:
            raise Exception(f'ERROR. Operation not recognized: {operation}')

    aprint('\tCopying to output...')
    arcpy.Raster(raster_processed).save(output)
    delete_fc(raster_processed)
    aprint(f'Raster saved to: {output}')
#checked
def create_database(input_gdb: str, background_gdb: str, output_dir: str, pressure_user_code: str, config_path: str, ha_user_code: str = None, start_year: int = None, end_year: int = None):
    """
    Arguments:
        input_gdb (str): a database containing input datasets
        background_gdb (str): a database with background data
        output_dir (str): name of output folder, where to export results
        pressure_user_code (str): code of the pressure
        config_path (str): path to configuration file
        ha_user_code (str): if given, only this human activity layer will be processed
        start_year (int): min of range of years between which to use data, if applicable
        end_year (int): max of range of years between which to use data, if applicable
    """
    timer = Timer()

    # create output directories
    os.makedirs(output_dir, exist_ok=True)
    out_dir_PL = os.path.join(output_dir, 'PL')  # pressure directory
    out_dir_HA = os.path.join(output_dir, 'HA')  # human activity directory
    os.makedirs(out_dir_PL, exist_ok=True)
    os.makedirs(out_dir_HA, exist_ok=True)

    # check input/output directories
    if not arcpy.Exists(input_gdb):
        arcpy.AddError(f'ERROR. File geodatabase with input data {input_gdb} does not exist.')
        return        
    if not arcpy.Exists(config_path):
        arcpy.AddError(f'ERROR. Configuration file {config_path} does not exist.')
        return
    if not arcpy.Exists(background_gdb):
        arcpy.AddError(f'ERROR. File geodatabase with background data {background_gdb} does not exist.')
        return
        
    # background dataset paths
    study_area = os.path.join(background_gdb, 'study_area')
    grid_vector = os.path.join(background_gdb, 'grid_vector')
    grid_raster = os.path.join(background_gdb, 'grid_raster')
    eez_borders = os.path.join(background_gdb, 'EEZ_and_land_borders_poly')
    rivers_polyline = os.path.join(background_gdb, 'rivers_polyline')
    
    # check background datasets
    if not arcpy.Exists(study_area):
        arcpy.AddError(f'ERROR. Study area background dataset {study_area} is used for calculation and does not exist.')
        return        
    if not arcpy.Exists(grid_vector):
        arcpy.AddError(f'ERROR. Vector grid background dataset {grid_vector} is used for calculation and does not exist.')
        return
    if not arcpy.Exists(grid_raster):
        arcpy.AddError(f'ERROR. Raster grid background dataset {grid_raster} is used for calculation and does not exist.')
        return
    
    # set the workspace
    work_gdb = arcpy.env.scratchGDB
    arcpy.env.workspace = work_gdb
    arcpy.env.overwriteOutput = True

    # set the project spatial reference
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(3035)

    # set processing extent
    arcpy.env.snapRaster = grid_raster
    arcpy.env.extent = grid_raster

    # load configuration file
    try:
        with open(config_path, 'rb') as f:
            config = toml.load(f)
        config = nan_to_none(config)    # process config file
    except Exception as e:
        arcpy.AddError(f'ERROR. Could not open configuration file {config_path}.')
        return
    
    try:

        pressure_codes = None
        if pressure_user_code == 'All':
            pressure_codes = list(config.keys())
        else:
            pressure_codes = [pressure_user_code]
            
        #####
        # Main loop
        #####

        for pressure_code in pressure_codes:
            pressure = config[pressure_code]
            pressure_name = pressure['name']
            pressure_post_processing = pressure['post_processing']
            pressure_rasterizing = pressure['rasterizing']
            pressure_output = os.path.join( out_dir_PL, pressure_code + '.tif' )    # output raster path
            aprint(f'----------\nCalculating human activities for pressure: {pressure_name} ({pressure_code})')

            activity_feature_classes = []
            activity_rasters = []
            
            # define human activities to process
            specific_ha_code = (ha_user_code != None and ha_user_code != '' and ha_user_code != 'All')
            if specific_ha_code:
                if ha_user_code in pressure.keys():
                    activities = {ha_user_code: pressure[ha_user_code]}
                else:
                    arcpy.AddError(f'ERROR. Human activity code {ha_user_code} is not present in config file for Pressure {pressure_code}.')
                    return
            else:
                activities = {activity_code: activity for activity_code, activity in pressure.items() if activity_code not in ['name', 'post_processing', 'rasterizing']}
            
            if pressure_code == 'PL_12':
                # for PL_12 two input layers are possible. If original input layer defined in config exists, then use that one, if not, use alternative one with suffix "eu".
                activity_codes = list(activities.keys())
                if len(activity_codes) > 1:
                    pass    # default behavior, individual activities already defined
                else:
                    use_original_layer = False
                    use_alternative_layer = False
                    
                    fish_input_original = os.path.join(input_gdb, pressure[activity_codes[0]]['input'] + '_polygon')
                    fish_input_alternative = os.path.join(input_gdb, pressure[activity_codes[0]]['input'] + '_eu_polygon')
                    if arcpy.Exists(fish_input_original):
                        if int(arcpy.management.GetCount(fish_input_original)[0]) > 0:
                            use_original_layer = True
                            pressure['useOriginal'] = True
                        else:
                            if arcpy.Exists(fish_input_alternative):
                                use_alternative_layer = True
                                pressure[activity_codes[0]]['input'] = pressure[activity_codes[0]]['input'] + '_eu'
                                pressure['useOriginal'] = False
                    else:
                        if arcpy.Exists(fish_input_alternative):
                            use_alternative_layer = True
                            pressure[activity_codes[0]]['input'] = pressure[activity_codes[0]]['input'] + '_eu'
                            pressure['useOriginal'] = False
                        else:
                            arcpy.AddError(f'ERROR. Neither {fish_input_original}, nor {fish_input_alternative} input dataset exist.')
                            delete_all_fc()
                            return
                
                    fish_layer = None
                    species_field = None
                    if use_original_layer:
                        fish_layer = fish_input_original
                        species_field = 'species'
                    elif use_alternative_layer:
                        fish_layer = fish_input_alternative
                        species_field = 'trgt_ss'
                        
                    if not field_exists(fish_layer, species_field):
                        arcpy.AddError(f'ERROR. Mandatory field "{species_field}" does not exist in the input dataset {fish_layer}.')
                        delete_all_fc()
                        return
                        
                    # fish activities defined by unique species in 'species' or 'trgt_ss' field, depends on input layer used
                    if use_original_layer or fish_input_alternative:
                        fish_species = [row[0] for row in arcpy.da.SearchCursor(fish_layer, species_field)]
                        fish_species = np.unique(np.array(fish_species))
                        activities = {activity_codes[0]+'_'+species: dict(activities[activity_codes[0]]) for species in fish_species}
                        for species in fish_species:
                            activities[activity_codes[0]+'_'+species]['query'] = f"{species_field} = '{species}'"
            
            layer_timer = Timer()

            # go through each human activity
            for activity_code, activity in activities.items():
                activity_name = activity['name']
                activity_query = activity['query']
                years_query_apply = activity['years']
                processing = activity['processing']
                post_processing = activity['post_processing']
                rasterizing = activity['rasterizing']
                activity_output = os.path.join( out_dir_HA, pressure_code + '_' + activity_code + '.tif' )    # output raster path

                bird_skip = False

                layer_timer.reset()
                aprint(f'---\nCalculating human activity: {activity_name} ({activity_code})')
                if len(processing) == 0:
                    arcpy.AddError(f'ERROR. No geometry and processing function is specified for input dataset used to calculate {activity_name} ({activity_code}) human activity dataset.')
                    delete_all_fc()
                    return

                layers = []     # all geometry layers of the activity
                # for every geometry type to be processed
                for instructions in processing:
                    geometry = instructions['geometry']
                    layer_name = activity['input'] + '_' + geometry
                    aprint(f'Processing input dataset: {layer_name}')
                    featureclass = os.path.join(input_gdb, layer_name)
                    if not arcpy.Exists(featureclass):
                        arcpy.AddError(f'ERROR. Input dataset {featureclass} is used to calculate {activity_name} dataset does not exist.')
                        delete_all_fc()
                        return
                    
                    # If query is applied and queried attribute does not exist - raise an error    
                    if activity_query is not None:
                        aprint(f'Applying query: {activity_query}')
                        field_name = activity_query.split()[0]
                        if not field_exists(featureclass, field_name):
                            arcpy.AddError(f'ERROR. The mandatory field "{field_name}" used to query input data does not exist in the input dataset {featureclass}.')
                            delete_all_fc()
                            return

                    # vector data
                    if geometry in ['point', 'polyline', 'polygon']:
                        is_raster = False

                        feature_num_total = int(arcpy.GetCount_management(featureclass)[0])
                        
                        # bird skip if origin field is not specified
                        if pressure_code == 'PL_15':
                            if not field_exists(featureclass, 'origin'):
                                arcpy.AddError(f'ERROR. Mandatory field "origin" does not exist in the input dataset {featureclass}.')
                                delete_all_fc()
                                return
                            bird_query, bird_query_layer = 'origin = NULL', 'bird_query_layer'
                            arcpy.MakeFeatureLayer_management(featureclass, bird_query_layer, bird_query)
                            if int(arcpy.GetCount_management(bird_query_layer)[0]) == feature_num_total:
                                bird_skip = True
                                activity_query = bird_query
                        
                        # dredging set unspecified fields to 'capital'
                        if activity_code == 'HA_05':
                            layer_modified = layer_name + '_modified'
                            arcpy.CopyFeatures_management(featureclass, layer_modified)
                            if not field_exists(featureclass, 'type'):
                                arcpy.AddError(f'ERROR. Mandatory field "type" does not exist in the input dataset {featureclass}.')
                                delete_all_fc()
                                return
                            with arcpy.da.UpdateCursor(featureclass, ['type']) as cursor:
                                for row in cursor:
                                    if row[0] == None:
                                        row[0] = 'capital'
                                    cursor.updateRow(row)
                            featureclass = layer_modified

                        
                        # apply query to the input dataset
                        temp_query = 'temp_query_' + activity_code + '_' + geometry
                        arcpy.MakeFeatureLayer_management(featureclass, temp_query, activity_query)
                                                
                        temp_layer = 'temp_' + activity_code + '_' + geometry
                        
                        # query layer by years if applicable
                        if years_query_apply and (start_year is not None or end_year is not None):
                            if field_exists(temp_query, 'year'):
                                year_query = None
                                if start_year is not None and end_year is not None:
                                    year_query = 'year >= ' + str(start_year) + ' AND year <= ' + str(end_year)
                                elif start_year is not None and end_year is None:
                                     year_query = 'year >= ' + str(start_year)
                                elif start_year is None and end_year is not None:
                                     year_query = 'year <= ' + str(end_year)
                            
                                temp_years = 'temp_years_' + activity_code + '_' + geometry
                                arcpy.MakeFeatureLayer_management(temp_query, temp_years, year_query)
                                
                                arcpy.CopyFeatures_management(temp_years, temp_layer)
                                delete_fc(temp_query)
                                delete_fc(temp_years)
                            else:
                                arcpy.AddWarning(f'Warning. Field "year" does not exist in the input dataset {featureclass}. Filtering by year can not to performed.')
                                arcpy.CopyFeatures_management(temp_query, temp_layer)
                                delete_fc(temp_query)
                        else:
                            arcpy.CopyFeatures_management(temp_query, temp_layer)
                            delete_fc(temp_query)

                        # verify that harbour points are not within harbour polygons
                        if activity_code == 'HA_10' and geometry == 'point':
                            polygon_layer = os.path.join(input_gdb, activity['input'] + '_polygon')
                            if not arcpy.Exists(polygon_layer):
                                arcpy.AddError(f'ERROR. Input dataset {polygon_layer} is used to calculate {activity_name} dataset does not exist.')
                                delete_all_fc()
                                return
                            temp_poly = 'temp_poly'
                            arcpy.MakeFeatureLayer_management(polygon_layer, temp_poly)
                            temp_point = 'temp_point'
                            arcpy.MakeFeatureLayer_management(temp_layer, temp_point)
                            arcpy.SelectLayerByLocation_management(temp_point, 'WITHIN', polygon_layer, '', 'NEW_SELECTION', 'INVERT')
                            temp_layer_reduced = temp_layer + '_reduced'
                            arcpy.CopyFeatures_management(temp_point, temp_layer_reduced)
                            arcpy.SelectLayerByAttribute_management(temp_point, selection_type='CLEAR_SELECTION')
                            for fc in [temp_layer, temp_poly, temp_point]:
                                delete_fc(fc)
                            temp_layer = temp_layer_reduced

                        # check if layer has no features
                        feature_num = int(arcpy.GetCount_management(temp_layer)[0])
                        if feature_num < 1:
                            if activity_query is None:
                                arcpy.AddWarning(f'WARNING. Dataset with geometry "{geometry}" is empty. Skipping this dataset for Pressure calculation.')
                            else:
                                arcpy.AddWarning(f'WARNING. Dataset with geometry "{geometry}" and applied query "{activity_query}" is empty. Skipping this dataset for Pressure calculation.')
                            continue
                            
                        try:
                            # process input feature class
                            if instructions != None:
                                if instructions['function'] == 'copy':
                                    temp_layer_processed = create_copy(temp_layer)
                                elif instructions['function'] == 'buffer':
                                    temp_layer_processed = create_buffer(temp_layer, instructions)
                                elif instructions['function'] == 'rings':
                                    temp_layer_processed = create_rings(temp_layer, instructions)
                                elif instructions['function'] == 'rivers':
                                    if not arcpy.Exists(rivers_polyline):
                                        arcpy.AddError(f'ERROR. Rivers background dataset {rivers_polyline} is used for calculation and does not exist.')
                                        delete_all_fc()
                                        return 
                                    temp_layer_processed = create_rivers(temp_layer, rivers_polyline, instructions)
                                elif instructions['function'] == 'spill':
                                    temp_layer_processed = create_spill(temp_layer, instructions, grid_vector)                     
                                elif instructions['function'] == 'birds':
                                    if not arcpy.Exists(eez_borders):
                                        arcpy.AddError(f'ERROR. EEZ borders background dataset {eez_borders} is used for calculation and does not exist.')
                                        delete_all_fc()
                                        return
                                    temp_layer_processed = create_birds(temp_layer, instructions, study_area, eez_borders)
                                elif instructions['function'] == 'mammals':
                                    if not arcpy.Exists(eez_borders):
                                        arcpy.AddError(f'ERROR. EEZ borders background dataset {eez_borders} is used for calculation and does not exist.')
                                        delete_all_fc()
                                        return
                                    temp_layer_processed = create_mammals(temp_layer, instructions, study_area, eez_borders)
                                elif instructions['function'] == 'fish':
                                    if pressure['useOriginal']:
                                        effort = os.path.join(input_gdb, instructions['effort'])
                                        if not arcpy.Exists(effort):
                                            arcpy.AddError(f'ERROR. Fishing effort dataset {effort} is used for calculation and does not exist.')
                                            delete_all_fc()
                                            return
                                        temp_layer_processed = create_fish_original(temp_layer, effort, instructions)
                                    else:
                                        temp_layer_processed = create_fish_alternative(temp_layer, instructions)
                                else:
                                    arcpy.AddError(f'ERROR. Unknown processing instructions given in config file for the input dataset {layer_name}.')
                                    delete_all_fc()
                                    return
                                delete_fc(temp_layer)
                            else:
                                arcpy.AddError(f'ERROR. No processing instructions given in config file for the input dataset {layer_name}.')
                                delete_all_fc()
                                return
                            if temp_layer_processed is not None:
                                layers.append(temp_layer_processed)
                        except Exception as e:
                            arcpy.AddError(f'An unhandled error occured when processing input dataset {layer_name}. Error message:')
                            exception_traceback(e)
                            delete_all_fc()
                            return

                    # raster data
                    elif geometry in ['raster', 'raster_tif']:
                        is_raster = True

                        # create temporary raster for operations
                        temp_raster = 'temp_' + activity_code + '_' + geometry
                        arcpy.CopyRaster_management(featureclass, temp_raster)

                        try:
                            # process input raster
                            if instructions != None:
                                if instructions['function'] == 'copy':
                                    aprint('Copying...')
                                    temp_raster_processed = temp_raster + '_copy'
                                    arcpy.CopyRaster_management(temp_raster, temp_raster_processed)
                                elif instructions['function'] == 'normalize':
                                    aprint('Normalizing...')
                                    temp_raster_processed = temp_raster + '_normalized'
                                    normalize_raster(temp_raster).save(temp_raster_processed)
                                elif instructions['function'] == 'rescale':
                                    aprint('Rescaling...')
                                    temp_raster_rescaled = temp_raster + '_rescale'
                                    arcpy.Resample_management(temp_raster, temp_raster_rescaled, grid_raster, "NEAREST")
                                    temp_raster_normalized = temp_raster + '_normalized'
                                    normalize_raster(temp_raster_rescaled).save(temp_raster_normalized)
                                    temp_raster_processed = temp_raster + '_clipped'
                                    arcpy.ia.Clip(temp_raster_normalized, study_area).save(temp_raster_processed)
                                    for fc in [temp_raster_rescaled, temp_raster_normalized]:
                                        delete_fc(fc)
                                else:
                                    arcpy.AddError(f'ERROR. Unknown processing instructions given in config file for the input dataset {layer_name}.')
                                    delete_all_fc()
                                    return
                                delete_fc(temp_raster)
                            else:
                                arcpy.AddError(f'ERROR. No processing instructions given in config file for the input dataset {layer_name}.')
                                delete_all_fc()
                                return
                            layers.append(temp_raster_processed)
                        except Exception as e:
                            arcpy.AddError(f'An unhandled error occured when processing input dataset {layer_name}. Error message:')
                            exception_traceback(e)
                            delete_all_fc()
                            return

                    else:
                        arcpy.AddError(f'ERROR. Unknown geometry type for input dataset {layer_name}')
                        delete_all_fc()
                        return
                
                if len(layers) < 1:
                    arcpy.AddWarning(f'WARNING. No input datasets processed for the human activity {activity_name} ({activity_code}). Human activity can not be calculated and will be not inluded for Pressure {pressure_name} ({pressure_code}).')
                    continue
                
                if is_raster:
                    if activity_code in ['HA_42']:
                        normalize_raster(layers[0]).save(activity_output)
                    else:
                        arcpy.CopyRaster_management(layers[0], activity_output)
                    aprint(f'Raster saved to: {activity_output}')
                    for layer in layers:
                        delete_fc(layer)
                else:
                    aprint('Merging feature classes...')
                    temp_layer_merged = 'temp_merged_' + activity_code
                    if len(layers) > 1:
                        arcpy.Merge_management(layers, temp_layer_merged)
                    else:
                        arcpy.CopyFeatures_management(layers[0], temp_layer_merged)
                    for layer in layers:
                        delete_fc(layer)

                    aprint('Postprocessing...')
                    temp_layer_processed = post_process(temp_layer_merged, post_processing, activity_code, study_area)
                    delete_fc(temp_layer_merged)

                    activity_feature_classes.append(temp_layer_processed)

                    # create output raster
                    create_raster(temp_layer_processed, rasterizing, activity_code, activity_output, grid_vector, grid_raster, None)

                activity_rasters.append(activity_output)

                aprint(f'Calculation of the human activity {activity_name} ({activity_code}) completed in {layer_timer.get_hhmmss()}')
                layer_timer.reset()

                # if origin was not specified, only one bird activity layer is produced
                if bird_skip:
                    break

            if not specific_ha_code:
                aprint(f'----------\nCalculating pressure: {pressure_name} ({pressure_code})')
                
                if len(activity_feature_classes) < 1:
                    if len(activity_rasters) < 1:
                        arcpy.AddWarning(f'WARNING. No human activities are created for pressure {pressure_name} ({pressure_code}). Pressure can not be calculated.')
                    else:
                        create_raster(None, pressure_rasterizing, pressure_code, pressure_output, grid_vector, grid_raster, activity_rasters)
                else:
                    # get unique geometry types
                    geometry_types = list(set([arcpy.Describe(x).shapeType for x in activity_feature_classes]))
                    if len(geometry_types) > 1:
                        activities_processed = None
                    else:
                        aprint('Merging feature classes...')
                        activities_merged = 'activities_merged_' + pressure_code
                        if len(activity_feature_classes) > 1:
                            arcpy.Merge_management(activity_feature_classes, activities_merged)
                        else:
                            arcpy.CopyFeatures_management(activity_feature_classes[0], activities_merged)
                        for layer in activity_feature_classes:
                            delete_fc(layer)

                        aprint('Postprocessing...')
                        activities_processed = post_process(activities_merged, pressure_post_processing, pressure_code, study_area)
                        delete_fc(activities_merged)

                    # create output raster
                    create_raster(activities_processed, pressure_rasterizing, pressure_code, pressure_output, grid_vector, grid_raster, activity_rasters)

                aprint(f'Calculation of pressure {pressure_name} ({pressure_code}) completed in {layer_timer.get_hhmmss()}')

    except Exception as e:
        arcpy.AddError(f'An unhandled error occured in the program. Error message:')
        exception_traceback(e)
        delete_all_fc()
        return

    # ensure all temporary layers are deleted
    delete_all_fc()
    
    aprint(f'Total execution time: {timer.get_duration()}')


if __name__ == "__main__":
    dt = datetime.datetime.now()
    aprint(f'--- Start processing at {dt.strftime("%Y %m %d %H:%M:%S")}')
    
    config_path = arcpy.GetParameterAsText(0)
    input_gdb = arcpy.GetParameterAsText(1)
    background_gdb = arcpy.GetParameterAsText(2)
    output_dir = arcpy.GetParameterAsText(3)
    pressure_label = arcpy.GetParameterAsText(4)
    pressure_code = pressure_label
    if ':' in pressure_label:
        pressure_code = pressure_label.split(':')[0]
    ha_label = arcpy.GetParameterAsText(5)
    ha_code = ha_label
    if ':' in ha_label:
        ha_code = ha_label.split(':')[0]
        arcpy.AddWarning('Pressure will not be calculated')
    start_year = arcpy.GetParameterAsText(6)
    start_year = int(start_year) if start_year != None and start_year != '' else None
    end_year = arcpy.GetParameterAsText(7)
    end_year = int(end_year) if end_year != None and end_year != '' else None
        
    create_database(input_gdb, background_gdb, output_dir, pressure_code, config_path, ha_code, start_year, end_year)
    
    dt = datetime.datetime.now()
    aprint(f'--- End processing at {dt.strftime("%Y %m %d %H:%M:%S")}')
