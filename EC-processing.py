
import datetime
import time
from collections import namedtuple
import arcpy
import arcpy.conversion
from arcpy.ia import *
from arcpy.sa import *
import tomli as toml
import os
import arcpy
import numpy as np
import traceback

is_test = True


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


def exception_traceback(e: Exception):
    """
    Format exception traceback and print
    """
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    aprint('--------------')
    aprint(''.join(tb))
    
def aprint(s: str):
    """
    Short for arcpy.AddMessage()
    """
    arcpy.AddMessage(s)

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
         
def delete_all_fc():
    """
    Delete all temporary datasets
    """
    try:
        for fc in arcpy.ListFeatureClasses() + arcpy.ListRasters():
            delete_fc(fc)
    except Exception as e:
        arcpy.AddWarning(f'WARNING. Can not delete temporary files.')

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
    
def create_copy_and_set_1(layer: str, attribute: str) -> str:
    """
    Creates a copy of the layer and sets value 1 if attribute value is Null

    Arguments:
        layer (str): feature class name
        attribute (str): attribute with final value
    
    Returns:
        layer_copy (str): feature class copy name
    """
    aprint('Copying...')
    layer_copy = layer + '_copy'
    arcpy.CopyFeatures_management(layer, layer_copy)
    
    # add EC value field to new layer
    attr = 'EC_value'
    arcpy.AddField_management(layer_copy, attr, 'FLOAT')
    
    with arcpy.da.UpdateCursor(in_table=layer_copy, field_names=[attribute, attr]) as cursor:
        for row in cursor:
            if row[0] is None:
                row[1] = 1
            else:
                row[1] = float(row[0])
            cursor.updateRow(row)
    return layer_copy

def process_benthic_species(layer: str, depth_value: int, bathymetry_raster: str) -> str:
    """
    Filter features that intersect bathymetry raster cells with value higher than idicated depth_value. Creates a copy of the layer with filtered features

    Arguments:
        layer (str): feature class name
        depth_value (int): depth value for filtering features
        bathymetry_raster (str): path to the raster with bathymetry data
    
    Returns:
        layer_filtered (str): filtered features feature class name
    """
    temp_points_bathymetry = 'temp_points_bathymetry'
    temp_reclasified_bathymetry = 'temp_reclasified_bathymetry'
    if not arcpy.Exists(temp_points_bathymetry):
        aprint('Reclassifying bathymetry raster...')
        min_value = arcpy.GetRasterProperties_management(bathymetry_raster, "MINIMUM").getOutput(0)
        reclassify_range = RemapRange([[min_value, depth_value-1, "NODATA"]])
        reclassify_raster = Reclassify(bathymetry_raster, "Value", reclassify_range)
        reclassify_raster.save(temp_reclasified_bathymetry)
        arcpy.conversion.RasterToPoint(reclassify_raster, temp_points_bathymetry, "VALUE")
    
    aprint('Intersecting with bathymetry...')
    temp_selectbyloc_layer = 'temp_selectbyloc_layer'
    arcpy.MakeFeatureLayer_management(layer, temp_selectbyloc_layer)
    arcpy.management.SelectLayerByLocation(temp_selectbyloc_layer, 'INTERSECT', temp_points_bathymetry)
    aprint('Copying...')
    layer_shallow = layer + '_shallow'
    arcpy.CopyFeatures_management(temp_selectbyloc_layer, layer_shallow)
    delete_fc(temp_selectbyloc_layer)
    return layer_shallow
    
def process_fish_species(layer: str, square_id_attribute: str, cpue_attribute: str, landings_attribute: str, year_attribute: str) -> str:
    """
    Filter features that intersect bathymetry raster cells with value higher than idicated depth_value. Creates a copy of the layer with filtered features

    Arguments:
        layer (str): feature class name
        square_id_attribute(str): attribute name
        cpue_attribute(str): attribute name 
        landings_attribute(str): attribute name 
    Returns:
        layer_filtered (str): feature class name
    """
    geometries = {}
    max_cpue = 0
    fields = [square_id_attribute, cpue_attribute, landings_attribute, year_attribute, 'SHAPE@', 'SHAPE@AREA']
    with arcpy.da.SearchCursor(layer, fields) as cursor:
        for row in cursor:
            sq_ID = row[0]
            cpue_value = row[1]
            landings_value = row[2]
            year = row[3]
            geom = row[4].WKT
            area = row[5] / 1000000 # area in square km
            
            # get max cpue value
            if cpue_value != None:
                if cpue_value > max_cpue:
                    max_cpue = cpue_value
            
            if sq_ID not in geometries:
                geometries[sq_ID] = {
                    'geom': geom,
                    'area': area
                }
            
            # for each unique square get cpue and landings values per year
            if year != None: 
                if year not in geometries[sq_ID]:
                    geometries[sq_ID][year] = {
                        'cpue': [],
                        'landings': []
                    }
                geometries[sq_ID][year]['cpue'].append(cpue_value)
                geometries[sq_ID][year]['landings'].append(landings_value)
    
    if max_cpue > 0:
        # calculate an average value of landings attribute per year per ICES square and get max avg
        sqid_landing_avgs = {}
        max_avg_landing = 0
        for sqid, value in geometries.items():
            annual_landings_sum = []
            for key, year_values in value.items():
                if key != 'geom' and key != 'area':
                    year_landings_sum = None
                    for val in year_values['landings']:
                        if val != None:
                            if year_landings_sum is None:
                                year_landings_sum = val
                            else:
                                year_landings_sum += val
                    annual_landings_sum.append(year_landings_sum)
            
            # if all years landing values are None for the square, then avg is None, 
            if all(v is None for v in annual_landings_sum):
                sqid_landing_avgs[sqid] = None
            # otherwise calculate all years avg
            else:
                annual_landings_sum_not_none = [v for v in annual_landings_sum if v is not None]
                sqid_landing_avgs[sqid] = sum(annual_landings_sum_not_none) / len(annual_landings_sum_not_none)
                # get max landing year avg
                if sqid_landing_avgs[sqid] > max_avg_landing:
                    max_avg_landing = sqid_landing_avgs[sqid]
                # divide avg by square area
                sqid_landing_avgs[sqid] = sqid_landing_avgs[sqid] / value['area']
                    
        layer_fish = layer + '_fish'
        attr = 'EC_value'
        arcpy.CreateFeatureclass_management(arcpy.Describe(layer).path, layer_fish, 'POLYGON')
        arcpy.AddField_management(layer_fish, attr, 'FLOAT')
        with arcpy.da.InsertCursor(layer_fish, ['SHAPE@', attr]) as cursor:
            # calculate an average value of cpue attribute per year per ICES square
            for sqid, value in geometries.items():
                annual_cpue_sum = []
                annual_landings_sum = []
                for key, year_values in value.items():
                    if key != 'geom' and key != 'area':
                        year_cpue_sum = None
                        for val in year_values['cpue']:
                            if val != None:
                                if year_cpue_sum is None:
                                    year_cpue_sum = val
                                else:
                                    year_cpue_sum += val
                        annual_cpue_sum.append(year_cpue_sum)
                        
                        year_landings_sum = None
                        for val in year_values['landings']:
                            if val != None:
                                if year_landings_sum is None:
                                    year_landings_sum = val
                                else:
                                    year_landings_sum += val
                        annual_landings_sum.append(year_landings_sum)
                
                polygon_value = None
                # if all years cpue values are None for the square, then calculte value based on method 2
                if all(v is None for v in annual_cpue_sum):
                    if sqid_landing_avgs[sqid] != None:
                        polygon_value = sqid_landing_avgs[sqid] / max_avg_landing * max_cpue
                        
                # otherwise value is all years cpue average  
                else:
                    annual_sum_not_none = [v for v in annual_cpue_sum if v is not None]
                    polygon_value = sum(annual_sum_not_none) / len(annual_sum_not_none)
                if polygon_value != None: 
                    cursor.insertRow([arcpy.FromWKT(value['geom']), polygon_value])
    else:
        arcpy.AddWarning(f'WARNING. All CPUE values are empty for the dataset. No processing can be performed.')
        return None
    
    return layer_fish
    
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

        if operation == 'clip':
            aprint('\tClipping to borders...')
            temp_clipped = 'temp_clipped_' + layer_code
            arcpy.Clip_analysis(layer_processed, study_area, temp_clipped)
            if layer_processed != layer: delete_fc(layer_processed)
            layer_processed = temp_clipped

        elif operation == '':
            aprint('\tWARNING. Empty operation given. Skipping...')
            pass

        else:
            raise Exception(f'ERROR. Operation not recognized: {operation}')
    
    if len(instructions) == 0:
        layer_processed += '_postprocessed'
        arcpy.CopyFeatures_management(layer, layer_processed)

    return layer_processed


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


def create_raster(layer: str, instructions: list, layer_code: str, output: str, grid_vector: str, grid_raster: str):
    """
    Rasterizes vector layer

    Arguments:
        layer (str): clipped feature class name
        instructions (list): operations to be performed to create output raster
        layer_code (str): unique layer identification code
        output (str): output raster name
        grid_vector (str): grid reference feature class
        grid_raster (str): grid reference raster
    """
    if arcpy.Exists(output):
        delete_fc(output)
        
    layer_processed = layer
    raster_processed = None
    
    for operation in instructions:

        if operation == 'copy':
            # make a copy of raster as the pressure output
            aprint('\tCopying raster...')
            temp_copy = 'temp_copy_' + layer_code
            arcpy.CopyRaster_management(raster_processed, temp_copy)
            raster_processed = temp_copy
            
        elif operation == 'presence_abscence':
            # convert polygon to raster; set 1 in the cells intersected by polygon; set 0 in the cell not intersected by polygon
            aprint('\tCalculating presence / abscence...')
            temp_grid = 'temp_grid_' + layer_code
            arcpy.CopyFeatures_management(grid_vector, temp_grid)
            arcpy.AddField_management(in_table=temp_grid, field_name='EC_value', field_type='SHORT')
            with arcpy.da.UpdateCursor(in_table=temp_grid, field_names=['EC_value']) as cursor:
                for row in cursor:
                    row[0] = 1
                    cursor.updateRow(row)
            temp_selected_cells = 'temp_selected_cells_' + layer_code
            arcpy.MakeFeatureLayer_management(temp_grid, temp_selected_cells)
            arcpy.SelectLayerByLocation_management(temp_selected_cells, 'INTERSECT', layer, selection_type='NEW_SELECTION')
            temp_raster_with_1 = 'temp_raster_with_1_' + layer_code
            arcpy.conversion.FeatureToRaster(in_features=temp_selected_cells, field='EC_value', out_raster=temp_raster_with_1, cell_size=grid_raster)
            rc = RasterCollection([grid_raster, temp_raster_with_1]) 
            raster_processed = Sum(rc, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=True)
            for fc in [temp_selected_cells, temp_raster_with_1, temp_grid]:
                delete_fc(fc)
                
        elif operation == 'set_value':
            # set EC_value from polygon to raster grids
            aprint('\tSetting polygon values to raster grids...')
            temp_grid = 'temp_grid_' + layer_code
            arcpy.CopyFeatures_management(grid_vector, temp_grid)
            temp_sp_join = 'temp_sp_join_' + layer_code
            arcpy.analysis.SpatialJoin(target_features=temp_grid, join_features=layer, out_feature_class=temp_sp_join, join_type='KEEP_COMMON', match_option='INTERSECT')
            temp_raster_with_values = 'temp_raster_with_values_' + layer_code
            arcpy.conversion.FeatureToRaster(in_features=temp_sp_join, field='EC_value', out_raster=temp_raster_with_values, cell_size=grid_raster)
            rc = RasterCollection([grid_raster, temp_raster_with_values]) 
            temp_raster_sum = Sum(rc, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=True)
            temp_raster_set_values = 'temp_raster_set_values' + layer_code
            temp_raster_sum.save(temp_raster_set_values)
            for fc in [temp_sp_join, temp_raster_with_values, temp_grid]:
                delete_fc(fc)
            raster_processed = temp_raster_set_values
            
        elif operation == 'normalize':
            aprint('\tNormalizing cell values...')
            temp_norm = 'temp_raster_normalized_' + layer_code
            normalize_raster(raster_processed).save(temp_norm)
            raster_processed = temp_norm
            
        else:
            raise Exception(f'ERROR. Operation not recognized: {operation}')

    aprint('\tCopying to output...')
    aprint(f'Raster saveing to: {output}')
    arcpy.Raster(raster_processed).save(output)
    delete_fc(raster_processed)
    aprint(f'Raster saved to: {output}')

def create_database(input_gdb: str, background_gdb: str, output_dir: str, ec_group_user: str, config_path: str, start_year: int = None, end_year: int = None):
    """
    Arguments:
        input_gdb (str): a database containing input datasets
        background_gdb (str): a database with background data
        output_dir (str): name of output folder, where to export results
        ec_group_user (str): group of the ecosystem component(s)
        config_path (str): path to configuration file
        start_year (int): min of range of years between which to use data, if applicable
        end_year (int): max of range of years between which to use data, if applicable
    """
    timer = Timer()
    
    # create output directories
    os.makedirs(output_dir, exist_ok=True)
    out_dir_EC = os.path.join(output_dir, 'EC')  # ecosystem component directory
    os.makedirs(out_dir_EC, exist_ok=True)

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
    bathymetry_raster = os.path.join(background_gdb, 'bathymetry')
    
    # check background datasets
    if not arcpy.Exists(study_area):
        arcpy.AddError(f'ERROR. Study area background dataset {study_area} does not exist.')
        return        
    if not arcpy.Exists(grid_vector):
        arcpy.AddError(f'ERROR. Vector grid background dataset {grid_vector} does not exist.')
        return
    if not arcpy.Exists(grid_raster):
        arcpy.AddError(f'ERROR. Raster grid background dataset {grid_raster} does not exist.')
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
    
        ec_groups = None
        if ec_group_user == 'All':
            ec_groups = list(config.keys())
        else:
            ec_groups = [ec_group_user]

        #####
        # Main loop
        #####

        for ec_group_name in ec_groups:
            ec_group = config[ec_group_name]
            ec_input = ec_group['input']
            years_query_apply = ec_group['years']
            ec_processing = ec_group['processing']
            ec_post_processing = ec_group['post_processing']
            ec_rasterizing = ec_group['rasterizing']
            aprint(f'----------\nProcessing ecosystem components group: {ec_group_name}')
            
            ec_feature_classes = []
            ec_rasters = []
            
            layer_timer = Timer()
            
            for ec_code, ec in ec_group.items():
                if ec_code.startswith('EC_'):                
                    ec_name = ec['name']
                    ec_query = ec['query']
                    ec_output = os.path.join( out_dir_EC, ec_code + '.tif' )    # output raster path
                    
                    layer_timer.reset()
                    aprint(f'---\nCalculating ecosystem component: {ec_name} ({ec_code})')
                                            
                    if len(ec_processing) == 0:
                        arcpy.AddError(f'ERROR. No geometry and processing function is specified in the config file for input dataset used to calculate {ec_name} ({ec_code}) ecosystem component dataset.')
                        delete_all_fc()
                        return
                    
                    layers = []     # all geometry layers of the ec
                    
                    for instructions in ec_processing:
                        geometry = instructions['geometry']
                        layer_name = ec_input + '_' + geometry
                        aprint(f'Processing input dataset: {layer_name}')
                        featureclass = os.path.join(input_gdb, layer_name)
                        if not arcpy.Exists(featureclass):
                            arcpy.AddError(f'ERROR. Input dataset {featureclass} used to calculate {ec_name} ({ec_code}) dataset does not exist.')
                            delete_all_fc()
                            return
                        
                        # If query is applied and queried attribute does not exist - raise an error    
                        if ec_query is not None:
                            aprint(f'Applying query: {ec_query}')
                            field_name = ec_query.split()[0]
                            if not field_exists(featureclass, field_name):
                                arcpy.AddError(f'ERROR. The mandatory field "{field_name}" specified in the config file used to query input data does not exist in the input dataset {featureclass}.')
                                delete_all_fc()
                                return
                        
                        # vector data
                        if geometry in ['point', 'polyline', 'polygon']:
                            is_raster = False
                            layer_name_ending = ec_code + '_' + geometry
                            
                            # apply query to the input dataset
                            temp_query = 'temp_query_' + layer_name_ending
                            arcpy.MakeFeatureLayer_management(featureclass, temp_query, ec_query)
                            
                            temp_layer = 'temp_' + layer_name_ending
                            
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
                                
                                    temp_years = 'temp_years_' + layer_name_ending
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
                            
                            # check if layer has no features
                            feature_num = int(arcpy.GetCount_management(temp_layer)[0])
                            if feature_num < 1:
                                if ec_query is None:
                                    arcpy.AddWarning(f'WARNING. Dataset with geometry "{geometry}" is empty. Skipping this dataset for Ecosystem component calculation.')
                                else:
                                    arcpy.AddWarning(f'WARNING. Dataset with geometry "{geometry}" and applied query "{ec_query}" is empty. Skipping this dataset for Ecosystem component calculation.')
                                continue

                            try:
                                # process input feature class
                                if instructions != None:
                                    if instructions['function'] == 'copy':
                                        temp_layer_processed = create_copy(temp_layer)
                                        if temp_layer_processed is not None:
                                            layers.append(temp_layer_processed)
                                    elif instructions['function'] == 'copy_and_set_1':
                                        temp_layer_processed = create_copy_and_set_1(temp_layer, instructions['attribute'])
                                        if temp_layer_processed is not None:
                                            layers.append(temp_layer_processed)
                                    elif instructions['function'] == 'process_benthic_species':
                                        if not arcpy.Exists(bathymetry_raster):
                                            arcpy.AddError(f'ERROR. Bathymetry raster background dataset {bathymetry_raster} does not exist.')
                                            delete_all_fc()
                                            return
                                        temp_layer_processed = process_benthic_species(temp_layer, instructions['depth_value'], bathymetry_raster)
                                        if temp_layer_processed is not None:
                                            layers.append(temp_layer_processed)
                                    elif instructions['function'] == 'process_fish_species':
                                        temp_layer_processed = process_fish_species(temp_layer, instructions['square_id_attribute'], instructions['cpue_attribute'], instructions['landings_attribute'], instructions['year_attribute'])
                                        if temp_layer_processed is not None:
                                            layers.append(temp_layer_processed)
                                    else:
                                        arcpy.AddError(f'ERROR. Unknown processing instructions given in config file for the input dataset {layer_name}.')
                                        delete_all_fc()
                                        return
                                    delete_fc(temp_layer)
                                else:
                                    arcpy.AddError(f'ERROR. No processing instructions given in config file for the input dataset {layer_name}.')
                                    delete_all_fc()
                                    return
                            except Exception as e:
                                arcpy.AddError(f'An unhandled error occured when processing input dataset {layer_name}. Error message:')
                                exception_traceback(e)
                                delete_all_fc()
                                return

                        else:
                            arcpy.AddError(f'ERROR. Unknown geometry type for input dataset {layer_name}')
                            delete_all_fc()
                            return
                    
                    if len(layers) > 0:
                        
                        aprint('Merging feature classes...')
                        temp_layer_merged = 'temp_merged_' + ec_code
                        if len(layers) > 1:
                            arcpy.Merge_management(layers, temp_layer_merged)
                        else:
                            arcpy.CopyFeatures_management(layers[0], temp_layer_merged)
                        for layer in layers:
                            delete_fc(layer)

                        aprint('Postprocessing...')
                        temp_layer_processed = post_process(temp_layer_merged, ec_post_processing, ec_code, study_area)
                        delete_fc(temp_layer_merged)

                        # create output raster
                        aprint('Creating raster...')
                        create_raster(temp_layer_processed, ec_rasterizing, ec_code, ec_output, grid_vector, grid_raster)

                        aprint(f'Ecosystem component {ec_name} ({ec_code}) completed in {layer_timer.get_hhmmss()}')
                        layer_timer.reset()

                        
                    else:
                        arcpy.AddWarning(f'WARNING. No input datasets processed for the ecosystem component {ec_name} ({ec_code}). No ecosystem component dataset created.')
                        aprint(f'Ecosystem component {ec_name} ({ec_code}) completed in {layer_timer.get_hhmmss()}')
                        layer_timer.reset()
                        
                    
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
    ec_group = arcpy.GetParameterAsText(4)
    start_year = arcpy.GetParameterAsText(5)
    start_year = int(start_year) if start_year != None and start_year != '' else None
    end_year = arcpy.GetParameterAsText(6)
    end_year = int(end_year) if end_year != None and end_year != '' else None
        
    create_database(input_gdb, background_gdb, output_dir, ec_group, config_path, start_year, end_year)
    
    dt = datetime.datetime.now()
    aprint(f'--- End processing at {dt.strftime("%Y %m %d %H:%M:%S")}')
